//! Rewrites a Baseline stream's order counts from type 2 to type 1, for a
//! test of type 1: no encoder at hand writes it (x264 writes types 0 and
//! 2). Only the SPS changes. Its `pic_order_cnt_type` 2 becomes 1 with
//! `delta_pic_order_always_zero_flag` set, `offset_for_non_ref_pic` 1,
//! `offset_for_top_to_bottom_field` 0 and one `offset_for_ref_frame` of
//! 2. That gives every frame the count type 2 gave it (8.2.1.2 against
//! 8.2.1.3), and with the always-zero flag the slice headers carry no
//! order count fields under either type, so they stay as they are.
//!
//! ```sh
//! cargo run --example h264-poc1 -- in.264 out.264
//! ```
//!
//! Takes an Annex B stream whose SPSs are Baseline, Main or Extended (no
//! chroma fields before the order count type) with type 2.

fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [input, output] = args.as_slice() else {
        eprintln!("usage: h264-poc1 <in.264> <out.264>");
        return std::process::ExitCode::FAILURE;
    };
    match convert(input, output) {
        Ok(count) => {
            eprintln!("rewrote {count} SPS");
            std::process::ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("h264-poc1: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

fn convert(input: &str, output: &str) -> Result<usize, String> {
    use reims_vgpu_video::codec::h264::nal::{AnnexB, NalUnit, nal_type};
    let data = std::fs::read(input).map_err(|e| format!("{input}: {e}"))?;
    let mut out = Vec::with_capacity(data.len() + 64);
    let mut count = 0;
    for unit in AnnexB::new(&data) {
        let nal = NalUnit::new(unit).map_err(|e| e.to_string())?;
        out.extend_from_slice(&[0, 0, 0, 1]);
        if nal.unit_type() == nal_type::SPS {
            out.push(unit[0]);
            out.extend(escape(&rewrite_sps(&unescape(&unit[1..]))?));
            count += 1;
        } else {
            out.extend_from_slice(unit);
        }
    }
    std::fs::write(output, out).map_err(|e| format!("{output}: {e}"))?;
    Ok(count)
}

fn unescape(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut zeros = 0;
    for &b in data {
        if zeros >= 2 && b == 3 {
            zeros = 0;
            continue;
        }
        zeros = if b == 0 { zeros + 1 } else { 0 };
        out.push(b);
    }
    out
}

fn escape(rbsp: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rbsp.len() + 8);
    let mut zeros = 0;
    for &b in rbsp {
        if zeros >= 2 && b <= 3 {
            out.push(3);
            zeros = 0;
        }
        zeros = if b == 0 { zeros + 1 } else { 0 };
        out.push(b);
    }
    out
}

struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
}

impl Bits<'_> {
    fn bit(&mut self) -> Result<bool, String> {
        let byte = self.data.get(self.pos / 8).ok_or("SPS ends early")?;
        let bit = (byte >> (7 - self.pos % 8)) & 1 == 1;
        self.pos += 1;
        Ok(bit)
    }

    fn ue(&mut self) -> Result<u32, String> {
        let mut zeros = 0;
        while !self.bit()? {
            zeros += 1;
            if zeros > 31 {
                return Err("bad Exp-Golomb code".into());
            }
        }
        let mut value = 1u64;
        for _ in 0..zeros {
            value = (value << 1) | u64::from(self.bit()?);
        }
        u32::try_from(value - 1).map_err(|e| e.to_string())
    }
}

struct Writer {
    bits: Vec<bool>,
}

impl Writer {
    fn ue(&mut self, value: u32) {
        let code = u64::from(value) + 1;
        let len = 64 - code.leading_zeros();
        self.bits
            .extend(std::iter::repeat_n(false, len as usize - 1));
        for i in (0..len).rev() {
            self.bits.push((code >> i) & 1 == 1);
        }
    }

    fn se(&mut self, value: i32) {
        let code = if value > 0 {
            2 * value.unsigned_abs() - 1
        } else {
            2 * value.unsigned_abs()
        };
        self.ue(code);
    }
}

fn rewrite_sps(rbsp: &[u8]) -> Result<Vec<u8>, String> {
    let mut r = Bits { data: rbsp, pos: 0 };
    let profile_idc = rbsp.first().copied().ok_or("empty SPS")?;
    if !matches!(profile_idc, 66 | 77 | 88) {
        return Err(format!(
            "profile_idc {profile_idc} codes chroma fields first"
        ));
    }
    r.pos = 24;
    r.ue()?; // seq_parameter_set_id
    r.ue()?; // log2_max_frame_num_minus4
    let start = r.pos;
    if r.ue()? != 2 {
        return Err("pic_order_cnt_type is not 2".into());
    }
    let end = r.pos;
    // The stop bit: the last set bit of the RBSP.
    let last = rbsp
        .iter()
        .rposition(|&b| b != 0)
        .ok_or("SPS has no stop bit")?;
    let stop = last * 8 + 7 - rbsp[last].trailing_zeros() as usize;
    let bit_at = |i: usize| (rbsp[i / 8] >> (7 - i % 8)) & 1 == 1;
    let mut w = Writer {
        bits: (0..start).map(bit_at).collect(),
    };
    w.ue(1); // pic_order_cnt_type
    w.bits.push(true); // delta_pic_order_always_zero_flag
    w.se(1); // offset_for_non_ref_pic
    w.se(0); // offset_for_top_to_bottom_field
    w.ue(1); // num_ref_frames_in_pic_order_cnt_cycle
    w.se(2); // offset_for_ref_frame[0]
    w.bits.extend((end..stop).map(bit_at));
    w.bits.push(true); // rbsp_stop_one_bit
    while !w.bits.len().is_multiple_of(8) {
        w.bits.push(false);
    }
    Ok(w.bits
        .chunks(8)
        .map(|byte| byte.iter().fold(0u8, |acc, &b| (acc << 1) | u8::from(b)))
        .collect())
}
