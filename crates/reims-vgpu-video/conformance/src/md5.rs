//! MD5 (RFC 1321), to compare frames with a reference decoder's
//! `framemd5` output without a dependency.

/// An MD5 digest being computed.
pub struct Md5 {
    table: [u32; 64],
    state: [u32; 4],
    block: [u8; 64],
    filled: usize,
    length: u64,
}

/// The per-round shift amounts (RFC 1321, 3.4).
const SHIFTS: [u32; 16] = [7, 12, 17, 22, 5, 9, 14, 20, 4, 11, 16, 23, 6, 10, 15, 21];

fn sine_table() -> [u32; 64] {
    // RFC 1321, 3.4: T[i] is the integer part of 4294967296 * abs(sin(i)).
    // An f64 holds each exactly enough for its 32 integer bits.
    core::array::from_fn(|i| ((i as f64 + 1.0).sin().abs() * 4_294_967_296.0) as u32)
}

impl Default for Md5 {
    fn default() -> Self {
        Self::new()
    }
}

impl Md5 {
    /// A digest of nothing yet.
    pub fn new() -> Self {
        Self {
            table: sine_table(),
            state: [0x6745_2301, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476],
            block: [0; 64],
            filled: 0,
            length: 0,
        }
    }

    /// Takes in `data`, after what came before.
    pub fn update(&mut self, mut data: &[u8]) {
        self.length = self.length.wrapping_add(data.len() as u64);
        while !data.is_empty() {
            let take = (64 - self.filled).min(data.len());
            self.block[self.filled..self.filled + take].copy_from_slice(&data[..take]);
            self.filled += take;
            data = &data[take..];
            if self.filled == 64 {
                let block = self.block;
                self.compress(&block);
                self.filled = 0;
            }
        }
    }

    /// The digest, in lowercase hex, as `md5sum` and ffmpeg's `framemd5`
    /// print it.
    pub fn hex(mut self) -> String {
        let bits = self.length.wrapping_mul(8);
        self.update(&[0x80]);
        while self.filled != 56 {
            self.update(&[0]);
        }
        self.update(&bits.to_le_bytes());
        self.state
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .map(|b| format!("{b:02x}"))
            .collect()
    }

    fn compress(&mut self, block: &[u8; 64]) {
        let t = &self.table;
        let m: [u32; 16] = core::array::from_fn(|i| {
            u32::from_le_bytes([
                block[4 * i],
                block[4 * i + 1],
                block[4 * i + 2],
                block[4 * i + 3],
            ])
        });
        let [mut a, mut b, mut c, mut d] = self.state;
        for i in 0..64 {
            let (f, g) = match i / 16 {
                0 => ((b & c) | (!b & d), i),
                1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                2 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            let rotated = a
                .wrapping_add(f)
                .wrapping_add(t[i])
                .wrapping_add(m[g])
                .rotate_left(SHIFTS[(i / 16) * 4 + i % 4]);
            a = d;
            d = c;
            c = b;
            b = b.wrapping_add(rotated);
        }
        for (s, v) in self.state.iter_mut().zip([a, b, c, d]) {
            *s = s.wrapping_add(v);
        }
    }
}

/// Whether the digest gives RFC 1321's test suite's answers, A.5, in part.
pub fn self_test() -> bool {
    let hex = |s: &[u8]| {
        let mut h = Md5::new();
        h.update(s);
        h.hex()
    };
    hex(b"") == "d41d8cd98f00b204e9800998ecf8427e"
        && hex(b"abc") == "900150983cd24fb0d6963f7d28e17f72"
        && hex(b"12345678901234567890123456789012345678901234567890123456789012345678901234567890")
            == "57edf4a22be3c955ac49da2e2107b67a"
}
