//! The draws' check at a video's open on a Windows host, a diagnostic (`vaapi`'s
//! `rt_windows`): its pictures, expectations and comparison, which build and run on every host.
//! It changes nothing: the video draws its pictures whatever the check reads (`graphics::drawn`).
//! It is to be removed by the commit that lands the fix of the cause it is to find.
//!
//! A picture of known samples ([`picture`]), 8-bit and then 10-bit, is drawn into a pair of
//! targets twice: whole ([`CASES`]' first), then from an offset and short of the targets, which
//! owe zeros first (`PlaneTargets::owe_zeros`), so that the draw clears them and every texel past
//! the picture is to read zero. After each draw each plane is read twice: by the Direct3D 11
//! device that made the targets (`PlaneDraw::read_back`), and by the device through the texture
//! of the plane's format that the guest's textures over the plane take, sampled. Each read is
//! compared with [`expected`] exactly ([`compare`]), and the open writes one line of what each
//! read, with what the driver answers of rt's import of the targets.
//!
//! **The reads of real pictures** (`rt_windows`' module doc) are compared here too: once a boot
//! for each of the first four sizes and depths of the sessions' pictures and each of a session's
//! [`READ_PICTURES`] ([`takes`]), four reads of one picture, each of its planes against another
//! texel by texel ([`differ`]), and the decoded luma's figures ([`luma_figures`]), in one line
//! ([`reads_line`]).

/// The check's pictures' size in luma samples: the chroma's is half, 32 by 32.
pub(crate) const SIZE: (u32, u32) = (64, 64);

/// One draw of the check: from `origin` (the crop's first column and row, even) `extent` luma
/// samples, into targets that owe zeros first where `owe_zeros`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Case {
    pub(crate) origin: (u32, u32),
    pub(crate) extent: (u32, u32),
    pub(crate) owe_zeros: bool,
    /// The case in the log line.
    name: &'static str,
}

/// The check's draws into one pair of targets, in order (the module doc).
pub(crate) const CASES: [Case; 2] = [
    Case {
        origin: (0, 0),
        extent: SIZE,
        owe_zeros: false,
        name: "drawn whole",
    },
    Case {
        origin: (16, 8),
        extent: (40, 30),
        owe_zeros: true,
        name: "drawn short over zeros",
    },
];

/// Sample `channel` of plane `plane` (0 luma, 1 chroma: Cb 0, Cr 1) at `(x, y)` of the check's
/// picture: 8 bits, or 10 bits in the top of 16 as P010 holds them.
fn sample(ten_bit: bool, plane: u32, channel: u32, (x, y): (u32, u32)) -> u16 {
    let v = x * 29 + y * 71 + x * y + channel * 113 + plane * 57;
    let v = if ten_bit { (v & 0x3ff) << 6 } else { v & 0xff };
    u16::try_from(v).unwrap_or(u16::MAX)
}

/// The size of plane `plane` of the check's picture.
fn plane_size(plane: usize) -> (u32, u32) {
    match plane {
        0 => SIZE,
        _ => (SIZE.0.div_ceil(2), SIZE.1.div_ceil(2)),
    }
}

/// The check's picture's luma and chroma bytes, row after row with nothing between them,
/// little-endian: one byte a sample, or two where `ten_bit`, the chroma's Cb then Cr.
pub(crate) fn picture(ten_bit: bool) -> [Vec<u8>; 2] {
    std::array::from_fn(|plane| {
        let (width, height) = plane_size(plane);
        let channels = if plane == 0 { 1 } else { 2 };
        let mut bytes = Vec::new();
        for y in 0..height {
            for x in 0..width {
                for channel in 0..channels {
                    let plane = u32::try_from(plane).unwrap_or(1);
                    let [low, high] = sample(ten_bit, plane, channel, (x, y)).to_le_bytes();
                    bytes.push(low);
                    if ten_bit {
                        bytes.push(high);
                    }
                }
            }
        }
        bytes
    })
}

/// What plane `plane` is to read after `case`'s draw of the check's picture: by texel, row after
/// row, its samples (the luma's in the first, its second 0), the picture's from the draw's origin
/// inside its extent (half of each in the chroma, rounded up), and 0 past it.
pub(crate) fn expected(ten_bit: bool, case: &Case, plane: usize) -> Vec<[u16; 2]> {
    let (width, height) = plane_size(plane);
    let (origin, extent) = match plane {
        0 => (case.origin, case.extent),
        _ => (
            (case.origin.0 / 2, case.origin.1 / 2),
            (case.extent.0.div_ceil(2), case.extent.1.div_ceil(2)),
        ),
    };
    let index = u32::try_from(plane).unwrap_or(1);
    let mut texels = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let texel = if x < extent.0 && y < extent.1 {
                let at = (x + origin.0, y + origin.1);
                let second = if plane == 0 {
                    0
                } else {
                    sample(ten_bit, index, 1, at)
                };
                [sample(ten_bit, index, 0, at), second]
            } else {
                [0, 0]
            };
            texels.push(texel);
        }
    }
    texels
}

/// What `got`, plane `plane` read in its samples (the luma's in the first), is against
/// `expected`'s texels, exactly: `right`, or how many texels are wrong, of how many, and the
/// first, with what it was to read and read.
pub(crate) fn compare(plane: usize, expected: &[[u16; 2]], got: &[[f32; 2]]) -> String {
    let channels = if plane == 0 { 1 } else { 2 };
    let width = usize::try_from(plane_size(plane).0).unwrap_or(1);
    let mut first = None;
    let mut wrong = 0;
    for (i, want) in expected.iter().enumerate() {
        let texel = got.get(i).copied().unwrap_or([f32::NAN; 2]);
        if (0..channels).any(|c| texel[c] != f32::from(want[c])) {
            wrong += 1;
            first.get_or_insert((i, *want, texel));
        }
    }
    let Some((at, want, texel)) = first else {
        return "right".to_owned();
    };
    format!(
        "{wrong} of {} wrong, the first at ({}, {}): expected {:?}, got {:?}",
        expected.len(),
        at % width,
        at / width,
        &want[..channels],
        &texel[..channels],
    )
}

/// Plane `plane`'s texels in `bytes` as the Direct3D 11 device reads them back
/// (`PlaneDraw::read_back`), in their samples: a byte a sample, or two little-endian ones where
/// `ten_bit`, the chroma's Cb then Cr, as [`picture`] lays them.
pub(crate) fn from_bytes(ten_bit: bool, plane: usize, bytes: &[u8]) -> Vec<[f32; 2]> {
    let size = if ten_bit { 2 } else { 1 };
    let channels = if plane == 0 { 1 } else { 2 };
    let sample = |s: &[u8]| match s {
        [low, high] => f32::from(u16::from_le_bytes([*low, *high])),
        [one] => f32::from(*one),
        _ => f32::NAN,
    };
    (bytes.chunks_exact(size * channels))
        .map(|texel| {
            let mut samples = texel.chunks_exact(size).map(sample);
            let first = samples.next().unwrap_or(f32::NAN);
            [first, samples.next().unwrap_or(0.0)]
        })
        .collect()
}

/// Plane `plane`'s texels as the device's sampling reads them (`read`, each texel's channels
/// as floats), in their samples: each channel times the format's largest value, rounded; the
/// luma's second 0.
pub(crate) fn from_floats(ten_bit: bool, plane: usize, read: &[[f32; 4]]) -> Vec<[f32; 2]> {
    let largest = if ten_bit { 65535.0 } else { 255.0 };
    (read.iter())
        .map(|texel| {
            let second = if plane == 0 {
                0.0
            } else {
                (texel[1] * largest).round()
            };
            [(texel[0] * largest).round(), second]
        })
        .collect()
}

impl Case {
    /// The case's name in the line.
    pub(crate) fn name(&self) -> &'static str {
        self.name
    }
}

/// The pictures of a session the reads of real pictures read, by their place among its pictures
/// drawn: its first and its 30th.
pub(crate) const READ_PICTURES: [u64; 2] = [1, 30];

/// The most sizes and depths of pictures a boot's reads of real pictures read.
const READ_KEYS: usize = 4;

/// What a session's pictures are to the reads of real pictures: their crop's width and height,
/// and whether their samples are 10-bit.
pub(crate) type ReadKey = ((u32, u32), bool);

/// Whether picture `READ_PICTURES[which]` of a session of pictures of `key` is read, `read` being
/// the boot's keys so far, each with whether each of [`READ_PICTURES`] was read: once a boot for
/// each key and picture, for the first [`READ_KEYS`] keys, so that at most eight are read and a
/// session too short to reach the 30th leaves it to the next of its key. Marks it read.
pub(crate) fn takes(read: &mut Vec<(ReadKey, [bool; 2])>, key: ReadKey, which: usize) -> bool {
    let at = match read.iter().position(|(k, _)| *k == key) {
        Some(at) => at,
        None if read.len() < READ_KEYS => {
            read.push((key, [false; 2]));
            read.len() - 1
        }
        None => return false,
    };
    let slot = (read.get_mut(at)).and_then(|(_, taken)| taken.get_mut(which));
    slot.is_some_and(|slot| !std::mem::replace(slot, true))
}

/// One read of a picture's two planes, the luma's and the chroma's: each plane's bytes, row after
/// row with nothing between them, and its width in texels; `Err` says why it read none.
pub(crate) type Read = Result<[(Vec<u8>, usize); 2], String>;

/// One plane as a read holds it: its texels row after row with nothing between them, `width` a
/// row, of `texel` bytes each (at most 4).
#[derive(Clone, Copy)]
struct PlaneRead<'a> {
    bytes: &'a [u8],
    width: usize,
    texel: usize,
}

impl PlaneRead<'_> {
    /// Its whole rows.
    fn height(&self) -> usize {
        (self.bytes.len())
            .checked_div(self.width * self.texel)
            .unwrap_or(0)
    }

    /// The texel at (`x`, `y`), its bytes as a little-endian number.
    fn texel(&self, x: usize, y: usize) -> u32 {
        let at = (y * self.width + x) * self.texel;
        let mut word = [0; 4];
        if let Some(bytes) = self.bytes.get(at..at + self.texel) {
            word.iter_mut()
                .zip(bytes)
                .for_each(|(to, from)| *to = *from);
        }
        u32::from_le_bytes(word)
    }
}

/// How `a` and `b` differ, texel by texel over the columns and rows both hold: how many texels
/// differ, of how many, their bounding box, and the first eight's columns and rows with `a`'s
/// value and `b`'s.
fn differ(a: PlaneRead<'_>, b: PlaneRead<'_>) -> String {
    let (width, height) = (a.width.min(b.width), a.height().min(b.height()));
    let (mut count, mut bounds, mut first) = (0, None, Vec::new());
    for y in 0..height {
        for x in 0..width {
            let (one, other) = (a.texel(x, y), b.texel(x, y));
            if one == other {
                continue;
            }
            count += 1;
            let [x0, _, x1, y1] = bounds.get_or_insert([x, y, x, y]);
            (*x0, *x1, *y1) = ((*x0).min(x), (*x1).max(x), y);
            if first.len() < 8 {
                first.push(format!("({x}, {y}) {one:#x}/{other:#x}"));
            }
        }
    }
    let total = width * height;
    match bounds {
        None => format!("0 of {total} differ"),
        Some([x0, y0, x1, y1]) => format!(
            "{count} of {total} differ, in ({x0}, {y0})-({x1}, {y1}), first {}",
            first.join(", ")
        ),
    }
}

/// The figures of a decoded luma plane's samples in `bytes`, a byte each, or two little-endian
/// ones where `ten_bit` (P010's 10 bits in the top of 16, as it stores them): the least, the most,
/// the mean and how many are zero, of how many.
fn luma_figures(bytes: &[u8], ten_bit: bool) -> String {
    let size = if ten_bit { 2 } else { 1 };
    let (mut least, mut most, mut sum, mut zeros, mut n) = (u16::MAX, 0, 0, 0, 0);
    for sample in bytes.chunks_exact(size) {
        let value = match sample {
            [low, high] => u16::from_le_bytes([*low, *high]),
            [one] => u16::from(*one),
            _ => 0,
        };
        (least, most) = (least.min(value), most.max(value));
        sum += u64::from(value);
        zeros += u64::from(value == 0);
        n += 1u64;
    }
    let Some(tenths) = (sum * 10).checked_div(n) else {
        return "no samples".to_owned();
    };
    format!(
        "min {least} max {most} mean {}.{} zeros {zeros} of {n}",
        tenths / 10,
        tenths % 10
    )
}

/// Plane `plane` (0 the luma, 1 the chroma) of `read`, its samples 16 bits where `ten_bit`, else
/// 8; `Err` says why `which` read it not.
fn plane_of<'a>(
    read: &'a Read,
    which: &str,
    plane: usize,
    ten_bit: bool,
) -> Result<PlaneRead<'a>, String> {
    let sample = if ten_bit { 2 } else { 1 };
    match read.as_ref().map(|planes| planes.get(plane)) {
        Ok(Some((bytes, width))) => Ok(PlaneRead {
            bytes,
            width: *width,
            texel: sample * (1 + plane),
        }),
        Ok(None) => Err(format!("{which} holds no plane {plane}")),
        Err(why) => Err(format!("{which} not read ({why})")),
    }
}

/// The line of the reads of one real picture: `picture`, what it is, then for each plane how the
/// `decoded` read differs from `d3d`'s, `d3d`'s from rt's `before`, and `before` from rt's
/// `after`, each plane's samples 16 bits where `ten_bit`, else 8, and the decoded luma's figures.
pub(crate) fn reads_line(
    picture: &str,
    ten_bit: bool,
    [decoded, d3d, before, after]: [&Read; 4],
) -> String {
    let pair = |(a, one): (&Read, &str), (b, other): (&Read, &str), at: usize| {
        let a = plane_of(a, one, at, ten_bit);
        let b = plane_of(b, other, at, ten_bit);
        let differs = match (a, b) {
            (Ok(a), Ok(b)) => differ(a, b),
            (Err(why), _) | (_, Err(why)) => why,
        };
        format!("{one} against {other}: {differs}")
    };
    let side = |at: usize| {
        let reads = [
            (decoded, "decode"),
            (d3d, "Direct3D"),
            (before, "rt"),
            (after, "rt again"),
        ];
        let lines: Vec<String> = reads.windows(2).map(|w| pair(w[0], w[1], at)).collect();
        lines.join(", ")
    };
    let figures = match decoded {
        Ok([(luma, _), _]) => luma_figures(luma, ten_bit),
        Err(why) => format!("not read ({why})"),
    };
    format!(
        "{picture}; luma: {}; chroma: {}; the decoded luma: {figures}",
        side(0),
        side(1)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `expected`'s texels as the device reads them through a plane's texture: each sample over
    /// the format's largest value, the luma's second channel 0.
    fn read_right(ten_bit: bool, expected: &[[u16; 2]]) -> Vec<[f32; 4]> {
        let largest = if ten_bit { 65535.0 } else { 255.0 };
        let read = |s: u16| f32::from(s) / largest;
        (expected.iter())
            .map(|&[a, b]| [read(a), read(b), 0.0, 1.0])
            .collect()
    }

    /// The pictures hold each plane's samples, little-endian, 10-bit ones in the top bits of 16,
    /// and the whole draw is to read them back as they are; the short draw reads the picture
    /// from its offset inside its extent and zeros past it.
    #[test]
    fn the_expectations_are_the_pictures_samples_and_zeros_past_a_short_draw() {
        for ten_bit in [false, true] {
            let [luma, chroma] = picture(ten_bit);
            let bytes = if ten_bit { 2 } else { 1 };
            assert_eq!(luma.len(), 64 * 64 * bytes);
            assert_eq!(chroma.len(), 32 * 32 * 2 * bytes);
            let whole = expected(ten_bit, &CASES[0], 1);
            let at = 5 * 32 + 3;
            let pair = &chroma[at * 2 * bytes..(at + 1) * 2 * bytes];
            let first = if ten_bit {
                u16::from_le_bytes([pair[0], pair[1]])
            } else {
                u16::from(pair[0])
            };
            assert_eq!(whole[at][0], first);
            if ten_bit {
                assert!(luma.chunks(2).all(|s| s[0] & 0x3f == 0));
            }
            let short = expected(ten_bit, &CASES[1], 0);
            assert_eq!(short[0], [sample(ten_bit, 0, 0, (16, 8)), 0]);
            assert_eq!(short[29 * 64 + 39], [sample(ten_bit, 0, 0, (55, 37)), 0]);
            assert_eq!(short[29 * 64 + 40], [0, 0]);
            assert_eq!(short[30 * 64], [0, 0]);
            let short = expected(ten_bit, &CASES[1], 1);
            assert_eq!(short[14 * 32 + 19][1], sample(ten_bit, 1, 1, (27, 18)));
            assert_eq!(short[15 * 32], [0, 0]);
        }
    }

    /// A plane read exactly, by either side, is right; one read a step off names how many
    /// texels are wrong and the first, with what it was to read and read.
    #[test]
    fn a_plane_read_exactly_is_right_and_one_texel_off_names_the_count_and_the_first() {
        for (ten_bit, plane) in [(false, 0), (false, 1), (true, 0), (true, 1)] {
            let want = expected(ten_bit, &CASES[1], plane);
            let mut read = read_right(ten_bit, &want);
            let got = from_floats(ten_bit, plane, &read);
            assert_eq!(compare(plane, &want, &got), "right");
            let largest = if ten_bit { 65535.0 } else { 255.0 };
            let at = 3 * usize::try_from(plane_size(plane).0).unwrap() + 7;
            read[at][plane] += 1.0 / largest;
            read[at + 1][plane] += 1.0 / largest;
            let line = compare(plane, &want, &from_floats(ten_bit, plane, &read));
            let total = want.len();
            assert!(
                line.starts_with(&format!(
                    "2 of {total} wrong, the first at (7, 3): expected {:?}, got ",
                    &want[at][..plane + 1]
                )),
                "{line}"
            );
        }
    }

    /// The Direct3D 11 device's read back of a whole draw holds the picture's bytes as they
    /// are, so the picture's own bytes read as the whole case's expectation.
    #[test]
    fn the_pictures_bytes_read_as_the_whole_draws_expectation() {
        for ten_bit in [false, true] {
            let planes = picture(ten_bit);
            for (plane, bytes) in planes.iter().enumerate() {
                let want = expected(ten_bit, &CASES[0], plane);
                let got = from_bytes(ten_bit, plane, bytes);
                assert_eq!(compare(plane, &want, &got), "right", "{ten_bit} {plane}");
            }
            let mut bytes = planes[1].clone();
            bytes[0] ^= 1;
            let got = from_bytes(ten_bit, 1, &bytes);
            assert!(compare(1, &expected(ten_bit, &CASES[0], 1), &got).starts_with("1 of 1024"));
        }
    }

    /// A boot reads each of a session's two pictures once for each of the first four sizes and
    /// depths of pictures, none of a fifth's, and a session too short to reach its 30th leaves it
    /// to the next of its size and depth.
    #[test]
    fn a_boot_reads_each_picture_once_for_four_sizes_and_depths() {
        let mut read = Vec::new();
        let key = |n: u32, ten_bit: bool| ((1920 + n, 1080), ten_bit);
        assert!(takes(&mut read, key(0, false), 0));
        assert!(!takes(&mut read, key(0, false), 0));
        assert!(takes(&mut read, key(0, false), 1));
        assert!(!takes(&mut read, key(0, false), 1));
        assert!(takes(&mut read, key(0, true), 0));
        assert!(takes(&mut read, key(1, false), 0));
        assert!(takes(&mut read, key(2, false), 0));
        assert!(!takes(&mut read, key(3, false), 0));
        assert!(!takes(&mut read, key(3, false), 1));
        assert!(takes(&mut read, key(2, false), 1));
        assert_eq!(read.len(), 4);
    }

    /// Two reads of a plane differ by how many texels differ, of those both hold, their bounding
    /// box and the first eight with both values; equal ones by none.
    #[test]
    fn two_reads_differ_by_the_count_the_box_and_the_first_eight() {
        fn plane(bytes: &[u8]) -> PlaneRead<'_> {
            PlaneRead {
                bytes,
                width: 4,
                texel: 1,
            }
        }
        let zeros = [0u8; 16];
        let mut ones = [1u8; 16];
        ones[0] = 0;
        assert_eq!(differ(plane(&zeros), plane(&zeros)), "0 of 16 differ");
        assert_eq!(
            differ(plane(&zeros), plane(&ones)),
            "15 of 16 differ, in (0, 0)-(3, 3), first (1, 0) 0x0/0x1, (2, 0) 0x0/0x1, \
             (3, 0) 0x0/0x1, (0, 1) 0x0/0x1, (1, 1) 0x0/0x1, (2, 1) 0x0/0x1, (3, 1) 0x0/0x1, \
             (0, 2) 0x0/0x1"
        );
    }

    /// The line names the picture, then for each plane the decode against Direct3D, Direct3D
    /// against rt and rt against itself again, over the texels both reads hold, a read that was
    /// not made in its place, and the decoded luma's figures.
    #[test]
    fn the_reads_line_names_each_planes_differences_and_the_luma_figures() {
        let decoded: Read = Ok([((16..24).collect(), 4), (vec![128, 129, 130, 131], 2)]);
        let targets = |odd: u8, past: u8| -> Read {
            let luma = [16, 17, 18, 19, 0, 0, 20, odd, 22, 23, 0, 0];
            Ok([
                (luma.to_vec(), 6),
                (vec![128, 129, 130, 131, past, past], 3),
            ])
        };
        let (d3d, rt) = (targets(21, 0), targets(0x99, 0xff));
        let again: Read = Err("no reader".to_owned());
        let line = reads_line("a picture", false, [&decoded, &d3d, &rt, &again]);
        assert_eq!(
            line,
            "a picture; luma: decode against Direct3D: 0 of 8 differ, Direct3D against rt: 1 of \
             12 differ, in (1, 1)-(1, 1), first (1, 1) 0x15/0x99, rt against rt again: rt again \
             not read (no reader); chroma: decode against Direct3D: 0 of 2 differ, Direct3D \
             against rt: 1 of 3 differ, in (2, 0)-(2, 0), first (2, 0) 0x0/0xffff, rt against \
             rt again: rt again not read (no reader); the decoded luma: min 16 max 23 mean 19.5 \
             zeros 0 of 8"
        );
    }

    /// The decoded luma's figures read 10-bit samples as P010 stores them, and count its zeros.
    #[test]
    fn the_luma_figures_read_ten_bit_samples_as_stored() {
        let bytes = [0x00, 0x10, 0x00, 0x20, 0x00, 0x00];
        assert_eq!(
            luma_figures(&bytes, true),
            "min 0 max 8192 mean 4096.0 zeros 1 of 3"
        );
        assert_eq!(luma_figures(&[], false), "no samples");
    }
}
