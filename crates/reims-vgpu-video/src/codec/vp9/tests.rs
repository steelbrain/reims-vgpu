//! VP9's superframes, uncompressed headers and decoding process, on frames
//! written here from the specification's syntax and an accelerator that
//! records what it is asked to decode, and on the streams libvpx made for
//! the hardware tests (`tests/`, with their reference frames' MD5s). The
//! compressed headers and tiles are placeholders: nothing here decodes
//! samples.

mod fuzz;

use std::sync::Arc;

use super::config::VpCodecConfiguration;
use super::header::{self, FrameHeader, State};
use super::quant::{self, SegmentParams};
use super::superframe;
use super::*;

/// Profile 0's stream: 640x360, 30 frames of ffmpeg's `testsrc2` coded by
/// libvpx-vp9 (two passes, altref frames on, so three superframes of a
/// hidden frame and a shown one), in IVF.
pub(crate) const PROFILE_0_IVF: &[u8] = include_bytes!("tests/vp9p0.ivf");
/// Profile 2's, at 10 bits, made the same way.
pub(crate) const PROFILE_2_IVF: &[u8] = include_bytes!("tests/vp9p2.ivf");
/// The MD5 of each frame of [`PROFILE_0_IVF`] as libvpx's `vpxdec` decodes
/// it, over its NV12 planes (luma, then interleaved Cb and Cr), one a line,
/// as the hardware tests read their frames back. ffmpeg's own VP9 decoder
/// gives the same frames.
pub(crate) const PROFILE_0_MD5: &str = include_str!("tests/vp9p0.md5");
/// Likewise for [`PROFILE_2_IVF`], over P010 planes: each sample in the
/// high bits of a 16-bit little-endian word.
pub(crate) const PROFILE_2_MD5: &str = include_str!("tests/vp9p2.md5");

/// An IVF file's frames: each frame record's bytes after its 12-byte
/// header.
pub(crate) fn ivf_frames(file: &[u8]) -> Vec<&[u8]> {
    assert_eq!(&file[..4], b"DKIF", "an IVF file");
    let header = usize::from(u16::from_le_bytes([file[6], file[7]]));
    let mut frames = Vec::new();
    let mut at = header;
    while at + 12 <= file.len() {
        let size = u32::from_le_bytes(file[at..at + 4].try_into().unwrap()) as usize;
        frames.push(&file[at + 12..at + 12 + size]);
        at += 12 + size;
    }
    frames
}

/// Writes syntax elements, most significant bit first.
#[derive(Default)]
pub(crate) struct BitWriter {
    bits: Vec<bool>,
}

impl BitWriter {
    pub(crate) fn f(&mut self, n: u32, value: u32) -> &mut Self {
        for i in (0..n).rev() {
            self.bits.push((value >> i) & 1 == 1);
        }
        self
    }

    pub(crate) fn bit(&mut self, value: bool) -> &mut Self {
        self.bits.push(value);
        self
    }

    /// `su(n)`.
    pub(crate) fn su(&mut self, n: u32, value: i32) -> &mut Self {
        self.f(n, value.unsigned_abs()).bit(value < 0)
    }

    /// The bits, zero-padded to a byte.
    pub(crate) fn bytes(&self) -> Vec<u8> {
        self.bits
            .chunks(8)
            .map(|byte| {
                byte.iter()
                    .enumerate()
                    .fold(0u8, |acc, (i, &bit)| acc | (u8::from(bit) << (7 - i)))
            })
            .collect()
    }
}

/// Each reference's and mode's loop filter delta a frame updates, `None`
/// for one it leaves.
pub(crate) type Deltas = ([Option<i32>; 4], [Option<i32>; 2]);

/// The tree probabilities a segmentation map update codes, `None` for one
/// left at 255, and the prediction probabilities where it is temporal.
pub(crate) type MapProbs = ([Option<u32>; 7], Option<[Option<u32>; 3]>);

/// A recorded frame: its picture, header, the slots' pictures and sizes,
/// and segment 0's derived values.
pub(crate) type Recorded = (
    u32,
    FrameHeader,
    [Option<(u32, u32, u32)>; 8],
    SegmentParams,
);

/// What a frame's uncompressed header says, for [`FrameSpec::write`].
#[derive(Clone)]
pub(crate) struct FrameSpec {
    pub(crate) profile: u32,
    pub(crate) show_existing: Option<u32>,
    pub(crate) key: bool,
    pub(crate) show: bool,
    pub(crate) error_resilient: bool,
    pub(crate) intra_only: bool,
    pub(crate) reset_frame_context: u32,
    /// Profile 2's `ten_or_twelve_bit`.
    pub(crate) twelve_bit: bool,
    pub(crate) color_space: u32,
    pub(crate) full_range: bool,
    /// Profiles 1 and 3: `subsampling_x`, `subsampling_y`.
    pub(crate) subsampling: (bool, bool),
    pub(crate) refresh: u32,
    pub(crate) refs: [u32; 3],
    pub(crate) sign_bias: [bool; 3],
    /// Where an inter frame takes its size from a reference: its position
    /// among `refs`; `None` codes the size.
    pub(crate) size_from_ref: Option<usize>,
    pub(crate) size: (u32, u32),
    pub(crate) render: Option<(u32, u32)>,
    pub(crate) high_precision_mv: bool,
    /// `None` is switchable; else the coded literal.
    pub(crate) filter: Option<u32>,
    pub(crate) refresh_context: bool,
    pub(crate) parallel: bool,
    pub(crate) context_idx: u32,
    pub(crate) filter_level: u32,
    pub(crate) sharpness: u32,
    /// `loop_filter_delta_enabled`, and the deltas updated: each
    /// reference's and mode's, `None` for one not updated.
    pub(crate) deltas: Option<Deltas>,
    pub(crate) base_q: u32,
    pub(crate) delta_q: [i32; 3],
    pub(crate) segmentation: Option<SegmentationSpec>,
    /// Tile column increments coded (each a 1), then a 0 where room is
    /// left; and `tile_rows_log2`.
    pub(crate) tile_cols_increments: u32,
    pub(crate) tile_rows: u32,
    /// `header_size_in_bytes`; the frame then has that many bytes of
    /// compressed header and `tile_bytes` of tiles.
    pub(crate) compressed: u32,
    pub(crate) tile_bytes: usize,
}

/// What `segmentation_params()` codes of an enabled segmentation.
#[derive(Clone, Default)]
pub(crate) struct SegmentationSpec {
    pub(crate) update_map: Option<MapProbs>,
    /// `segmentation_abs_or_delta_update`, and each segment's features:
    /// `None` not enabled.
    pub(crate) update_data: Option<(bool, [[Option<i32>; 4]; 8])>,
}

impl FrameSpec {
    /// A profile 0 key frame of `width` by `height`.
    pub(crate) fn key(width: u32, height: u32) -> Self {
        Self {
            profile: 0,
            show_existing: None,
            key: true,
            show: true,
            error_resilient: false,
            intra_only: false,
            reset_frame_context: 0,
            twelve_bit: false,
            color_space: 2,
            full_range: false,
            subsampling: (true, true),
            refresh: 0xff,
            refs: [0, 1, 2],
            sign_bias: [false; 3],
            size_from_ref: None,
            size: (width, height),
            render: None,
            high_precision_mv: false,
            filter: Some(1),
            refresh_context: true,
            parallel: false,
            context_idx: 0,
            filter_level: 10,
            sharpness: 0,
            deltas: Some(([None; 4], [None; 2])),
            base_q: 60,
            delta_q: [0; 3],
            segmentation: None,
            tile_cols_increments: 0,
            tile_rows: 0,
            compressed: 4,
            tile_bytes: 8,
        }
    }

    /// An inter frame of the same size, predicting from slots 0, 1 and 2,
    /// refreshing slot 0.
    pub(crate) fn inter(width: u32, height: u32) -> Self {
        Self {
            key: false,
            refresh: 1,
            size_from_ref: Some(0),
            ..Self::key(width, height)
        }
    }

    pub(crate) fn write(&self) -> Vec<u8> {
        let mut w = BitWriter::default();
        w.f(2, 2)
            .bit(self.profile & 1 == 1)
            .bit(self.profile >> 1 == 1);
        if self.profile == 3 {
            w.bit(false);
        }
        if let Some(index) = self.show_existing {
            w.bit(true).f(3, index);
            return w.bytes();
        }
        w.bit(false)
            .bit(!self.key)
            .bit(self.show)
            .bit(self.error_resilient);
        let color = |w: &mut BitWriter| {
            if self.profile >= 2 {
                w.bit(self.twelve_bit);
            }
            w.f(3, self.color_space);
            if self.color_space != 7 {
                w.bit(self.full_range);
                if self.profile == 1 || self.profile == 3 {
                    w.bit(self.subsampling.0).bit(self.subsampling.1).bit(false);
                }
            } else if self.profile == 1 || self.profile == 3 {
                w.bit(false);
            }
        };
        let size = |w: &mut BitWriter, (width, height): (u32, u32)| {
            w.f(16, width - 1).f(16, height - 1);
        };
        // render_size() follows the frame's size in every branch.
        let render = |w: &mut BitWriter| match self.render {
            Some((width, height)) => {
                w.bit(true).f(16, width - 1).f(16, height - 1);
            }
            None => {
                w.bit(false);
            }
        };
        let sync = |w: &mut BitWriter| {
            w.f(8, 0x49).f(8, 0x83).f(8, 0x42);
        };
        if self.key {
            sync(&mut w);
            color(&mut w);
            size(&mut w, self.size);
            render(&mut w);
        } else {
            if !self.show {
                w.bit(self.intra_only);
            }
            if !self.error_resilient {
                w.f(2, self.reset_frame_context);
            }
            if self.intra_only {
                sync(&mut w);
                if self.profile > 0 {
                    color(&mut w);
                }
                w.f(8, self.refresh);
                size(&mut w, self.size);
                render(&mut w);
            } else {
                w.f(8, self.refresh);
                for (index, bias) in self.refs.iter().zip(self.sign_bias) {
                    w.f(3, *index).bit(bias);
                }
                match self.size_from_ref {
                    Some(found) => {
                        for i in 0..=found {
                            w.bit(i == found);
                        }
                    }
                    None => {
                        w.bit(false).bit(false).bit(false);
                        size(&mut w, self.size);
                    }
                }
                render(&mut w);
            }
        }
        if !self.key && !self.intra_only {
            w.bit(self.high_precision_mv);
            match self.filter {
                None => w.bit(true),
                Some(literal) => w.bit(false).f(2, literal),
            };
        }
        if !self.error_resilient {
            w.bit(self.refresh_context).bit(self.parallel);
        }
        w.f(2, self.context_idx);
        w.f(6, self.filter_level).f(3, self.sharpness);
        match &self.deltas {
            None => {
                w.bit(false);
            }
            Some((refs, modes)) => {
                w.bit(true);
                let update = refs.iter().chain(modes.iter()).any(Option::is_some);
                w.bit(update);
                if update {
                    for delta in refs.iter().chain(modes.iter()) {
                        match delta {
                            Some(value) => w.bit(true).su(6, *value),
                            None => w.bit(false),
                        };
                    }
                }
            }
        }
        w.f(8, self.base_q);
        for delta in self.delta_q {
            if delta == 0 {
                w.bit(false);
            } else {
                w.bit(true).su(4, delta);
            }
        }
        match &self.segmentation {
            None => {
                w.bit(false);
            }
            Some(seg) => {
                w.bit(true);
                match &seg.update_map {
                    None => {
                        w.bit(false);
                    }
                    Some((tree, pred)) => {
                        w.bit(true);
                        let prob = |w: &mut BitWriter, p: &Option<u32>| match p {
                            Some(p) => {
                                w.bit(true).f(8, *p);
                            }
                            None => {
                                w.bit(false);
                            }
                        };
                        for p in tree {
                            prob(&mut w, p);
                        }
                        match pred {
                            None => {
                                w.bit(false);
                            }
                            Some(pred) => {
                                w.bit(true);
                                for p in pred {
                                    prob(&mut w, p);
                                }
                            }
                        }
                    }
                }
                match &seg.update_data {
                    None => {
                        w.bit(false);
                    }
                    Some((absolute, features)) => {
                        w.bit(true).bit(*absolute);
                        for segment in features {
                            for (level, feature) in segment.iter().enumerate() {
                                match feature {
                                    None => {
                                        w.bit(false);
                                    }
                                    Some(value) => {
                                        w.bit(true);
                                        let bits = [8, 6, 2, 0][level];
                                        w.f(bits, value.unsigned_abs());
                                        if level < 2 {
                                            w.bit(*value < 0);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // tile_info(): the increments asked for, then a 0 where the
        // frame's width leaves room for more.
        let (min, max) = tile_bounds(self.size.0);
        for _ in 0..self.tile_cols_increments {
            w.bit(true);
        }
        if min + self.tile_cols_increments < max {
            w.bit(false);
        }
        match self.tile_rows {
            0 => w.bit(false),
            1 => w.bit(true).bit(false),
            _ => w.bit(true).bit(true),
        };
        w.f(16, self.compressed);
        let mut bytes = w.bytes();
        bytes.extend(std::iter::repeat_n(0xa5, self.compressed as usize));
        bytes.extend(std::iter::repeat_n(0x5a, self.tile_bytes));
        bytes
    }
}

/// `calc_min_log2_tile_cols()` and `calc_max_log2_tile_cols()` for a
/// frame `width` wide.
fn tile_bounds(width: u32) -> (u32, u32) {
    let sb64 = width.div_ceil(8).div_ceil(8);
    let mut min = 0;
    while (64 << min) < sb64 {
        min += 1;
    }
    let mut max = 1;
    while (sb64 >> max) >= 4 {
        max += 1;
    }
    (min, max - 1)
}

/// A superframe of `frames`, with sizes of `size_bytes` bytes.
pub(crate) fn superframe(frames: &[Vec<u8>], size_bytes: usize) -> Vec<u8> {
    let marker = 0xc0 | (((size_bytes - 1) as u8) << 3) | (frames.len() - 1) as u8;
    let mut out: Vec<u8> = frames.concat();
    out.push(marker);
    for frame in frames {
        out.extend_from_slice(&(frame.len() as u32).to_le_bytes()[..size_bytes]);
    }
    out.push(marker);
    out
}

/// An accelerator that records each frame it is asked to decode, and
/// fails where told to.
#[derive(Default)]
pub(crate) struct Recording {
    pub(crate) configured: Vec<Format>,
    pub(crate) next: u32,
    /// Each frame's picture, header, the slots' pictures, and segment 0's
    /// derived values.
    pub(crate) frames: Vec<Recorded>,
    pub(crate) fail_next: bool,
}

impl Accelerator for Recording {
    type Picture = u32;
    type Error = Error;

    fn configure(&mut self, format: &Format) -> Result<(), Error> {
        self.configured.push(*format);
        Ok(())
    }

    fn new_picture(&mut self) -> Result<u32, Error> {
        self.next += 1;
        Ok(self.next)
    }

    fn decode_frame(&mut self, picture: &u32, params: &FrameParams<'_, u32>) -> Result<(), Error> {
        if std::mem::take(&mut self.fail_next) {
            return Err(Error::Unsupported("the test's failure"));
        }
        let header = params.header;
        let uncompressed = usize::from(header.uncompressed_header_size);
        assert!(uncompressed + usize::from(header.header_size_in_bytes) <= params.frame.len());
        let slots = params
            .references
            .each_ref()
            .map(|r| r.as_ref().map(|r| (*r.picture, r.width, r.height)));
        self.frames
            .push((*picture, header.clone(), slots, params.segments[0]));
        Ok(())
    }
}

fn parse(frame: &[u8], state: &State) -> Result<(FrameHeader, State), Error> {
    header::parse(frame, state, &[Some((640, 360)); 8])
}

#[test]
fn a_sample_without_an_index_is_one_frame() {
    let frame = FrameSpec::key(64, 64).write();
    assert_eq!(superframe::frames(&frame).unwrap(), vec![&frame[..]]);
}

#[test]
fn a_superframe_splits_into_its_frames_at_every_size_width() {
    let a = FrameSpec::key(64, 64).write();
    let b = FrameSpec::inter(64, 64).write();
    for size_bytes in 1..=4 {
        let sample = superframe(&[a.clone(), b.clone()], size_bytes);
        let frames = superframe::frames(&sample).unwrap();
        assert_eq!(frames, vec![&a[..], &b[..]], "{size_bytes}-byte sizes");
    }
}

#[test]
fn hostile_superframe_indexes_are_refused_or_read_as_one_frame() {
    let a = FrameSpec::key(64, 64).write();
    // Sizes past the bytes before the index.
    let mut past = superframe(std::slice::from_ref(&a), 2);
    let at = past.len() - 3;
    past[at] = 0xff;
    past[at + 1] = 0xff;
    assert_eq!(superframe::frames(&past), Err(Error::BadSuperframe));
    // A frame of no bytes.
    let mut empty = superframe(&[a.clone(), vec![1]], 1);
    let at = empty.len() - 2;
    empty[at] = 0;
    assert_eq!(superframe::frames(&empty), Err(Error::BadSuperframe));
    // Four-byte sizes that overflow when added.
    let mut huge = superframe(&[a.clone(), a.clone()], 4);
    let index = huge.len() - 9;
    huge[index..index + 8].fill(0xff);
    assert_eq!(superframe::frames(&huge), Err(Error::BadSuperframe));
    // A marker with no twin where the index would begin: one frame.
    let mut lone = a.clone();
    lone.push(0xc1);
    assert_eq!(superframe::frames(&lone).unwrap(), vec![&lone[..]]);
    // A marker longer than the sample: one frame.
    assert_eq!(superframe::frames(&[0xdf]).unwrap(), vec![&[0xdf][..]]);
    assert_eq!(superframe::frames(&[]), Err(Error::EmptySample));
}

#[test]
fn every_truncation_of_a_superframe_is_refused_or_split_within_it() {
    let sample = superframe(
        &[
            FrameSpec::key(64, 64).write(),
            FrameSpec::inter(64, 64).write(),
        ],
        2,
    );
    for len in 0..sample.len() {
        let cut = &sample[..len];
        if let Ok(frames) = superframe::frames(cut) {
            let total: usize = frames.iter().map(|f| f.len()).sum();
            assert!(total <= len);
        }
    }
}

#[test]
fn a_key_frame_header_reads_as_written() {
    let mut spec = FrameSpec::key(1920, 1080);
    spec.render = Some((1920, 1072));
    spec.tile_cols_increments = 2;
    spec.tile_rows = 1;
    spec.delta_q = [-3, 0, 5];
    let frame = spec.write();
    let (header, state) = parse(&frame, &State::default()).unwrap();
    assert!(header.key_frame && header.show_frame && header.is_intra());
    assert_eq!(header.profile, 0);
    assert_eq!((header.width, header.height), (1920, 1080));
    assert_eq!((header.render_width, header.render_height), (1920, 1072));
    assert_eq!(header.refresh_frame_flags, 0xff);
    assert_eq!(header.color.bit_depth, 8);
    assert_eq!(header.color.color_space, 2);
    assert!(header.color.subsampling_x && header.color.subsampling_y);
    // 1920 is 30 superblocks: tile columns of 1 to 4 (log2 0 to 2).
    assert_eq!(header.tile_cols_log2, 2);
    assert_eq!(header.tile_rows_log2, 1);
    assert_eq!(header.quantization.base_q_idx, 60);
    assert_eq!(header.quantization.delta_q_y_dc, -3);
    assert_eq!(header.quantization.delta_q_uv_ac, 5);
    assert!(!header.quantization.lossless());
    assert_eq!(header.loop_filter.level, 10);
    assert_eq!(header.loop_filter.ref_deltas, [1, 0, -1, -1]);
    assert_eq!(header.header_size_in_bytes, 4);
    assert_eq!(
        usize::from(header.uncompressed_header_size) + 4 + 8,
        frame.len()
    );
    assert_eq!(state.color, Some(header.color));
}

#[test]
fn profile_2_reads_ten_and_twelve_bits() {
    let mut spec = FrameSpec::key(64, 64);
    spec.profile = 2;
    let (header, _) = parse(&spec.write(), &State::default()).unwrap();
    assert_eq!((header.profile, header.color.bit_depth), (2, 10));
    spec.twelve_bit = true;
    let (header, _) = parse(&spec.write(), &State::default()).unwrap();
    assert_eq!(header.color.bit_depth, 12);
}

#[test]
fn an_inter_frame_takes_its_size_from_the_reference_it_names() {
    let (_, state) = parse(&FrameSpec::key(640, 360).write(), &State::default()).unwrap();
    let mut spec = FrameSpec::inter(0, 0);
    spec.refs = [3, 5, 7];
    spec.size_from_ref = Some(1);
    spec.filter = None;
    spec.high_precision_mv = true;
    let mut sizes = [None; 8];
    sizes[5] = Some((320, 180));
    let (header, _) = header::parse(&spec.write(), &state, &sizes).unwrap();
    assert_eq!((header.width, header.height), (320, 180));
    assert_eq!(header.ref_frame_idx, [3, 5, 7]);
    assert_eq!(header.interp_filter, header::SWITCHABLE);
    assert!(header.allow_high_precision_mv);
    // An empty slot gives no size.
    sizes[5] = None;
    assert_eq!(
        header::parse(&spec.write(), &state, &sizes),
        Err(Error::MissingReference)
    );
}

#[test]
fn the_interpolation_filter_is_numbered_as_libvpx_numbers_it() {
    let (_, state) = parse(&FrameSpec::key(64, 64).write(), &State::default()).unwrap();
    // The coded literal 0 is EIGHTTAP_SMOOTH (1), 1 EIGHTTAP (0).
    for (literal, filter) in [(0, 1), (1, 0), (2, 2), (3, 3)] {
        let mut spec = FrameSpec::inter(64, 64);
        spec.filter = Some(literal);
        let (header, _) = parse(&spec.write(), &state).unwrap();
        assert_eq!(header.interp_filter, filter, "literal {literal}");
    }
}

#[test]
fn loop_filter_deltas_last_until_an_intra_frame_resets_them() {
    let mut key = FrameSpec::key(64, 64);
    key.deltas = Some(([None, Some(5), None, Some(-7)], [Some(3), None]));
    let (header, state) = parse(&key.write(), &State::default()).unwrap();
    assert_eq!(header.loop_filter.ref_deltas, [1, 5, -1, -7]);
    assert_eq!(header.loop_filter.mode_deltas, [3, 0]);
    // An inter frame that updates none keeps them.
    let mut inter = FrameSpec::inter(64, 64);
    inter.deltas = Some(([None; 4], [None; 2]));
    let (header, state) = parse(&inter.write(), &state).unwrap();
    assert_eq!(header.loop_filter.ref_deltas, [1, 5, -1, -7]);
    // A key frame starts them afresh.
    let (header, _) = parse(&FrameSpec::key(64, 64).write(), &state).unwrap();
    assert_eq!(header.loop_filter.ref_deltas, [1, 0, -1, -1]);
    assert_eq!(header.loop_filter.mode_deltas, [0, 0]);
}

#[test]
fn segmentation_features_last_until_an_intra_or_error_resilient_frame() {
    let mut features = [[None; 4]; 8];
    features[2] = [Some(-20), Some(9), Some(3), Some(0)];
    let mut key = FrameSpec::key(64, 64);
    key.segmentation = Some(SegmentationSpec {
        update_map: Some(([Some(10), None, None, None, None, None, Some(200)], None)),
        update_data: Some((false, features)),
    });
    let (header, state) = parse(&key.write(), &State::default()).unwrap();
    let seg = &header.segmentation;
    assert!(seg.enabled && seg.update_map && !seg.temporal_update);
    assert_eq!(seg.tree_probs, [10, 255, 255, 255, 255, 255, 200]);
    assert_eq!(seg.pred_probs, [255; 3]);
    assert_eq!(seg.feature_enabled[2], [true; 4]);
    assert_eq!(seg.feature_data[2], [-20, 9, 3, 0]);
    // An inter frame with segmentation on but nothing updated keeps all.
    let mut inter = FrameSpec::inter(64, 64);
    inter.segmentation = Some(SegmentationSpec::default());
    let (header, state) = parse(&inter.write(), &state).unwrap();
    let seg = &header.segmentation;
    assert!(seg.enabled && !seg.update_map);
    assert_eq!(seg.feature_data[2], [-20, 9, 3, 0]);
    assert_eq!(seg.tree_probs[6], 200);
    // One with segmentation off keeps them for later frames.
    let (header, state) = parse(&FrameSpec::inter(64, 64).write(), &state).unwrap();
    assert!(!header.segmentation.enabled);
    assert_eq!(state.segmentation.feature_data[2], [-20, 9, 3, 0]);
    // An error-resilient frame clears the features.
    let mut resilient = FrameSpec::inter(64, 64);
    resilient.error_resilient = true;
    let (_, state) = parse(&resilient.write(), &state).unwrap();
    assert_eq!(state.segmentation.feature_data[2], [0; 4]);
    assert_eq!(state.segmentation.feature_enabled[2], [false; 4]);
}

#[test]
fn an_intra_only_frame_of_profile_0_is_4_2_0_at_8_bits_and_names_its_slots() {
    let (_, state) = parse(&FrameSpec::key(64, 64).write(), &State::default()).unwrap();
    let mut spec = FrameSpec::key(128, 64);
    spec.key = false;
    spec.show = false;
    spec.intra_only = true;
    spec.refresh = 0b0000_0110;
    spec.context_idx = 3;
    let (header, _) = parse(&spec.write(), &state).unwrap();
    assert!(header.intra_only && header.is_intra() && !header.key_frame);
    assert_eq!(header.refresh_frame_flags, 0b110);
    assert_eq!(header.color.bit_depth, 8);
    assert_eq!(header.color.color_space, header::CS_BT_601);
    // libvpx and ffmpeg hand an intra frame's context index as 0.
    assert_eq!(header.frame_context_idx, 0);
}

#[test]
fn malformed_headers_are_refused() {
    let state = State::default();
    let good = FrameSpec::key(64, 64);
    // frame_marker other than 2.
    let mut bad = good.write();
    bad[0] ^= 0x80;
    assert_eq!(parse(&bad, &state), Err(Error::OutOfRange("frame_marker")));
    // A sync code off by a bit.
    let mut bad = good.write();
    bad[2] ^= 1;
    assert!(matches!(parse(&bad, &state), Err(Error::OutOfRange(_))));
    // RGB in profile 0.
    let mut rgb = good.clone();
    rgb.color_space = 7;
    assert_eq!(
        parse(&rgb.write(), &state),
        Err(Error::Unsupported("RGB in profile 0 or 2"))
    );
    // A compressed header past the frame, and an empty one.
    let mut past = good.clone();
    past.tile_bytes = 0;
    let mut frame = past.write();
    frame.pop();
    assert_eq!(parse(&frame, &state), Err(Error::BadHeaderSize));
    let mut empty = good.clone();
    empty.compressed = 0;
    assert_eq!(parse(&empty.write(), &state), Err(Error::BadHeaderSize));
    // An inter frame before any key frame.
    assert_eq!(
        parse(&FrameSpec::inter(64, 64).write(), &state),
        Err(Error::MissingReference)
    );
}

#[test]
fn every_truncation_of_a_header_is_refused() {
    let mut spec = FrameSpec::key(1920, 1080);
    spec.segmentation = Some(SegmentationSpec {
        update_map: Some(([Some(1); 7], Some([Some(2); 3]))),
        update_data: Some((true, [[Some(3); 4]; 8])),
    });
    spec.deltas = Some(([Some(-1); 4], [Some(2); 2]));
    let frame = spec.write();
    let (header, _) = parse(&frame, &State::default()).unwrap();
    let header_end = usize::from(header.uncompressed_header_size) + spec.compressed as usize;
    for len in 0..header_end {
        let result = parse(&frame[..len], &State::default());
        assert!(
            matches!(result, Err(Error::Truncated | Error::BadHeaderSize)),
            "{len} bytes: {result:?}"
        );
    }
}

#[test]
fn quantizer_scales_and_filter_levels_follow_the_segment_features() {
    let mut spec = FrameSpec::key(64, 64);
    spec.base_q = 0;
    let (header, _) = parse(&spec.write(), &State::default()).unwrap();
    let segments = quant::segment_params(&header);
    // The tables' first entries, at 8 bits.
    assert_eq!(
        (segments[0].luma_dc, segments[0].luma_ac),
        (4, 4),
        "q index 0"
    );
    assert!(header.quantization.lossless());
    let mut features = [[None; 4]; 8];
    features[1] = [Some(255), Some(63), None, None];
    spec.base_q = 100;
    spec.filter_level = 40;
    spec.deltas = Some(([None; 4], [Some(2), None]));
    spec.segmentation = Some(SegmentationSpec {
        update_map: None,
        update_data: Some((true, features)),
    });
    let (header, _) = parse(&spec.write(), &State::default()).unwrap();
    let segments = quant::segment_params(&header);
    // Segment 1 sets the index to 255 and the level to 63, absolutely.
    assert_eq!(segments[1].luma_ac, 1828);
    assert_eq!(segments[1].luma_dc, 1336);
    // Segment 0 keeps the frame's: level 40, shifted deltas (40 >= 32):
    // intra 1 << 1, last 0 + mode 2 << 1, golden (-1 + 0) << 1.
    assert_eq!(segments[0].filter_level[0], [42, 42]);
    assert_eq!(segments[0].filter_level[1], [44, 40]);
    assert_eq!(segments[0].filter_level[2], [42, 38]);
    assert_eq!(segments[1].filter_level[0], [63, 63]);
}

#[test]
fn a_vpcc_record_reads_and_its_streams_are_checked() {
    let p0 = VpCodecConfiguration::parse(&[1, 0, 0, 0, 0, 0x15, 0x82, 1, 1, 1, 0, 0]).unwrap();
    assert_eq!((p0.profile, p0.level, p0.bit_depth), (0, 21, 8));
    assert_eq!(p0.chroma_subsampling, 1);
    assert!(!p0.full_range && p0.is_4_2_0());
    assert_eq!(check_supported(&p0), Ok(()));
    let p2 = VpCodecConfiguration::parse(&[1, 0, 0, 0, 2, 0, 0xa3, 9, 16, 9, 0, 0]).unwrap();
    assert_eq!((p2.profile, p2.bit_depth), (2, 10));
    assert!(p2.full_range);
    assert_eq!(check_supported(&p2), Ok(()));
    let p1 = VpCodecConfiguration::parse(&[1, 0, 0, 0, 1, 0, 0x86, 1, 1, 1, 0, 0]).unwrap();
    assert_eq!(p1.chroma_subsampling, 3);
    assert!(matches!(check_supported(&p1), Err(Error::Unsupported(_))));
    for short in 0..12 {
        assert_eq!(
            VpCodecConfiguration::parse(&[1; 12][..short]),
            Err(Error::BadVpcc)
        );
    }
    assert_eq!(
        VpCodecConfiguration::parse(&[0, 0, 0, 0, 0, 0, 0x82, 1, 1, 1, 0, 0]),
        Err(Error::BadVpcc)
    );
}

#[test]
fn a_superframe_decodes_its_hidden_frame_into_its_slot_and_shows_the_other() {
    let mut decoder = Decoder::new(Recording::default());
    let key = decoder.decode(&FrameSpec::key(64, 64).write(), 0).unwrap();
    assert_eq!(key.len(), 1);
    assert!(key[0].key_frame);
    let mut hidden = FrameSpec::inter(64, 64);
    hidden.show = false;
    hidden.refresh = 1 << 6;
    let shown = FrameSpec::inter(64, 64);
    let sample = superframe(&[hidden.write(), shown.write()], 2);
    let out = decoder.decode(&sample, 1).unwrap();
    assert_eq!(out.len(), 1, "the shown frame alone");
    assert_eq!(*out[0].picture, 3);
    assert_eq!(out[0].id, 1);
    // The shown frame saw the hidden one in slot 6.
    let (_, _, slots, _) = decoder.accelerator().frames[2];
    assert_eq!(slots[6], Some((2, 64, 64)));
    assert_eq!(slots[1], Some((1, 64, 64)));
    // A frame that shows slot 6 shows the hidden frame's picture.
    let mut show = FrameSpec::key(64, 64);
    show.show_existing = Some(6);
    let out = decoder.decode(&show.write(), 2).unwrap();
    assert_eq!(*out[0].picture, 2);
    assert_eq!(decoder.accelerator().frames.len(), 3, "nothing decoded");
}

#[test]
fn a_stream_that_starts_without_a_key_frame_misses_its_references() {
    let mut decoder = Decoder::new(Recording::default());
    assert!(matches!(
        decoder.decode(&FrameSpec::inter(64, 64).write(), 0),
        Err(Error::MissingReference)
    ));
    let mut show = FrameSpec::key(64, 64);
    show.show_existing = Some(0);
    assert!(matches!(
        decoder.decode(&show.write(), 1),
        Err(Error::MissingReference)
    ));
    // A key frame then decodes.
    assert_eq!(
        decoder
            .decode(&FrameSpec::key(64, 64).write(), 2)
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn references_out_of_scale_or_format_are_refused() {
    let mut decoder = Decoder::new(Recording::default());
    decoder
        .decode(&FrameSpec::key(640, 360).write(), 0)
        .unwrap();
    // A frame more than twice as small as its reference.
    let mut small = FrameSpec::inter(300, 160);
    small.size_from_ref = None;
    assert!(matches!(
        decoder.decode(&small.write(), 1),
        Err(Error::BadReference)
    ));
    // Half the size scales.
    let mut half = FrameSpec::inter(320, 180);
    half.size_from_ref = None;
    half.refresh = 0;
    assert_eq!(decoder.decode(&half.write(), 2).unwrap().len(), 1);
    assert_eq!(
        decoder
            .accelerator()
            .configured
            .last()
            .map(|f| (f.width, f.height)),
        Some((320, 180)),
        "a frame of another size is configured for"
    );
    // A profile 2 key frame refills every slot at 10 bits; an 8-bit
    // intra-only frame of profile 0 then cannot be a reference for a
    // 10-bit frame.
    let mut p2 = FrameSpec::key(640, 360);
    p2.profile = 2;
    decoder.decode(&p2.write(), 3).unwrap();
    let mut intra = FrameSpec::key(640, 360);
    intra.key = false;
    intra.show = false;
    intra.intra_only = true;
    intra.refresh = 1;
    decoder.decode(&intra.write(), 4).unwrap();
    // The 8-bit intra frame set the colour configuration in force, so an
    // inter frame naming slot 1 (10-bit) is refused.
    let mut inter = FrameSpec::inter(640, 360);
    inter.refs = [1, 1, 1];
    assert!(matches!(
        decoder.decode(&inter.write(), 5),
        Err(Error::BadReference)
    ));
}

#[test]
fn unsupported_profiles_and_sizes_are_refused_before_the_accelerator() {
    let mut decoder = Decoder::new(Recording::default());
    let mut p1 = FrameSpec::key(64, 64);
    p1.profile = 1;
    p1.subsampling = (false, false);
    assert!(matches!(
        decoder.decode(&p1.write(), 0),
        Err(Error::Unsupported(_))
    ));
    let mut twelve = FrameSpec::key(64, 64);
    twelve.profile = 2;
    twelve.twelve_bit = true;
    assert!(matches!(
        decoder.decode(&twelve.write(), 1),
        Err(Error::Unsupported(_))
    ));
    assert!(matches!(
        decoder.decode(&FrameSpec::key(MAX_SIDE + 1, 64).write(), 2),
        Err(Error::TooLarge)
    ));
    assert!(decoder.accelerator().configured.is_empty());
}

#[test]
fn a_frame_that_does_not_decode_leaves_the_slots_and_state_as_they_were() {
    let mut decoder = Decoder::new(Recording::default());
    decoder.decode(&FrameSpec::key(64, 64).write(), 0).unwrap();
    let mut key = FrameSpec::key(64, 64);
    key.deltas = Some(([Some(9); 4], [None; 2]));
    decoder.accelerator_mut().fail_next = true;
    assert!(decoder.decode(&key.write(), 1).is_err());
    // The next inter frame still sees the first key frame in every slot,
    // and the deltas it left.
    let out = decoder
        .decode(&FrameSpec::inter(64, 64).write(), 2)
        .unwrap();
    assert_eq!(out.len(), 1);
    let (_, header, slots, _) = &decoder.accelerator().frames[1];
    assert_eq!(slots[0], Some((1, 64, 64)));
    assert_eq!(header.loop_filter.ref_deltas, [1, 0, -1, -1]);
}

#[test]
fn libvpx_streams_decode_every_frame_in_order() {
    for (file, profile, depth) in [(PROFILE_0_IVF, 0, 8), (PROFILE_2_IVF, 2, 10)] {
        let mut decoder = Decoder::new(Recording::default());
        let mut shown = 0;
        let mut superframes = 0;
        for (id, sample) in (0u64..).zip(ivf_frames(file)) {
            if superframe::frames(sample).unwrap().len() > 1 {
                superframes += 1;
            }
            let out = decoder.decode(sample, id).unwrap();
            assert_eq!(out.len(), 1, "sample {id} shows one frame");
            assert_eq!(out[0].size, (640, 360));
            assert_eq!(out[0].bit_depth, depth);
            shown += 1;
        }
        assert_eq!(shown, 30);
        assert_eq!(superframes, 3);
        let configured = &decoder.accelerator().configured;
        assert_eq!(configured.len(), 1);
        assert_eq!(configured[0].profile, profile);
        assert_eq!(decoder.accelerator().frames.len(), 33);
        let md5 = if profile == 0 {
            PROFILE_0_MD5
        } else {
            PROFILE_2_MD5
        };
        assert_eq!(md5.lines().count(), 30, "a reference MD5 a frame");
    }
}

#[test]
fn decoded_frames_keep_their_pictures_alive_beside_the_slots() {
    let mut decoder = Decoder::new(Recording::default());
    let out = decoder.decode(&FrameSpec::key(64, 64).write(), 0).unwrap();
    let picture = Arc::clone(&out[0].picture);
    // One count here, one in the output, eight slots.
    assert_eq!(Arc::strong_count(&picture), 10);
    decoder.reset();
    assert_eq!(Arc::strong_count(&picture), 2);
}
