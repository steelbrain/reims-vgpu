//! VP9 on NVDEC: the codec crate's decoder over an accelerator that fills
//! `CUVIDVP9PICPARAMS` from the frame's uncompressed header and the
//! reference slots, and hands NVDEC the frame whole, its uncompressed header
//! first, as one slice at its start, as ffmpeg's `nvdec_vp9.c` does. NVDEC
//! reads the compressed header and the tiles itself; NVIDIA's own parser is
//! not used. The frame is sent where it lies in the sample, with no copy.
//!
//! A decoder is made for each run of frames of one format and size (the
//! codec crate's `Accelerator::configure`), at the frame's size, as ffmpeg
//! makes one at a VP9 stream's `coded_width`, its frame's width. NVDEC names
//! a frame's references by their surfaces in its decoder, so an inter frame
//! whose references are another decoder's, a frame of another size than the
//! frames it predicts from, is refused as a picture's error, as are the
//! frames after it that predict from them, until an intra frame fills the
//! slots again (docs/video/bugs.md).

use std::ffi::{c_int, c_short, c_uint, c_ushort};
use std::sync::Arc;

use crate::codec::vp9::header::FrameHeader;
use crate::codec::vp9::{Accelerator, Format, FrameParams};

use super::decoder::{
    Decoder, DecoderSpec, InPlace, Picture, SPARE_SURFACES, Slices, active, surface_count,
};
use super::ffi::vp9::CUVIDVP9PICPARAMS;
use super::ffi::vp9::bits::{frame, segmentation};
use super::ffi::{self, Zeroed};
use super::{ChromaFormat, NvdecCodec};
use crate::nvidia::cuda::Gpu;
use crate::nvidia::error::Error;

/// VP9's reference slots.
const SLOTS: usize = 8;

/// A reference NVDEC is not to read: an intra frame's, as ffmpeg's
/// `ff_nvdec_get_ref_idx` gives a missing one.
const NO_REFERENCE: u8 = 0xff;

/// The NVDEC half of the VP9 decoder.
pub(crate) struct NvdecVp9 {
    gpu: Arc<Gpu>,
    decoder: Option<Decoder>,
    /// The frames sent: each in place, its own bytes of the sample.
    pub(crate) in_place: InPlace,
}

impl NvdecVp9 {
    pub(crate) fn new(gpu: Arc<Gpu>) -> Self {
        Self {
            gpu,
            decoder: None,
            in_place: InPlace::default(),
        }
    }
}

/// The decoder a run of `format`'s frames needs: 4:2:0 at its depth and
/// size, with a surface for each slot, the frame being decoded and the
/// spares.
pub(crate) fn spec(format: &Format) -> DecoderSpec {
    DecoderSpec {
        codec: NvdecCodec::Vp9,
        chroma: ChromaFormat::Yuv420,
        bit_depth: u32::from(format.bit_depth),
        width: format.width,
        height: format.height,
        surfaces: surface_count(SLOTS + 1 + SPARE_SURFACES),
    }
}

impl Accelerator for NvdecVp9 {
    type Picture = Picture;
    type Error = Error;

    fn configure(&mut self, format: &Format) -> Result<(), Error> {
        self.decoder = None;
        self.decoder = Some(Decoder::new(Arc::clone(&self.gpu), &spec(format))?);
        Ok(())
    }

    fn new_picture(&mut self) -> Result<Picture, Error> {
        active(self.decoder.as_ref())?.picture()
    }

    fn decode_frame(
        &mut self,
        picture: &Picture,
        params: &FrameParams<'_, Picture>,
    ) -> Result<(), Error> {
        let decoder = active(self.decoder.as_ref())?;
        let mut pic = picture_parameters(decoder, picture, params)?;
        // One slice, the frame whole, at its start.
        let decoded = decoder.decode(&mut pic, params.frame, Slices::Starts(&[0]));
        self.in_place.in_place += 1;
        decoded
    }
}

fn int(value: u32) -> c_int {
    c_int::try_from(value).unwrap_or(c_int::MAX)
}

/// The surface of the frame in `slot` that a frame of `decoder`'s names as
/// a reference.
fn reference(decoder: &Decoder, params: &FrameParams<'_, Picture>, slot: u8) -> Result<u8, Error> {
    let reference = (params.references.get(usize::from(slot)))
        .and_then(Option::as_ref)
        .ok_or_else(|| Error::Picture(format!("VP9: reference slot {slot} holds no frame")))?;
    if !reference.picture.is_of(decoder) {
        return Err(Error::Picture(format!(
            "VP9 on NVDEC: a {}x{} frame that predicts from a {}x{} one, of another decoder",
            params.header.width, params.header.height, reference.width, reference.height
        )));
    }
    u8::try_from(reference.picture.index())
        .map_err(|_| Error::Unsupported("a VP9 reference past surface 254".to_owned()))
}

/// The frame's `CUVIDPICPARAMS`, all but its bitstream, filled as ffmpeg's
/// `nvdec_vp9.c` fills them.
fn picture_parameters(
    decoder: &Decoder,
    picture: &Picture,
    p: &FrameParams<'_, Picture>,
) -> Result<ffi::CUVIDPICPARAMS, Error> {
    let h = p.header;
    let references = if h.is_intra() {
        [NO_REFERENCE; 3]
    } else {
        let [last, golden, alt] = h.ref_frame_idx;
        [
            reference(decoder, p, last)?,
            reference(decoder, p, golden)?,
            reference(decoder, p, alt)?,
        ]
    };
    let mut out = ffi::CUVIDPICPARAMS::zeroed();
    out.PicWidthInMbs = int(h.width.div_ceil(16));
    out.FrameHeightInMbs = int(h.height.div_ceil(16));
    out.CurrPicIdx = picture.index();
    out.CodecSpecific.vp9 = vp9_parameters(h, references);
    Ok(out)
}

/// `h`'s `CUVIDVP9PICPARAMS`, its last, golden and altref frames' surfaces
/// `references`.
fn vp9_parameters(h: &FrameHeader, references: [u8; 3]) -> CUVIDVP9PICPARAMS {
    let mut v = CUVIDVP9PICPARAMS::zeroed();
    v.width = h.width;
    v.height = h.height;
    [v.LastRefIdx, v.GoldenRefIdx, v.AltRefIdx] = references;
    // VP9's own `color_space`, which ffmpeg maps its colour spaces back to.
    v.colorSpace = h.color.color_space;
    let flag = u32::from;
    let frame_word = [
        (frame::profile, u32::from(h.profile)),
        (frame::frameContextIdx, u32::from(h.frame_context_idx)),
        // 0 a key frame, 1 another.
        (frame::frameType, flag(!h.key_frame)),
        (frame::showFrame, flag(h.show_frame)),
        (frame::errorResilient, flag(h.error_resilient_mode)),
        (
            frame::frameParallelDecoding,
            flag(h.frame_parallel_decoding_mode),
        ),
        (frame::subSamplingX, flag(h.color.subsampling_x)),
        (frame::subSamplingY, flag(h.color.subsampling_y)),
        (frame::intraOnly, flag(h.intra_only)),
        (
            frame::allow_high_precision_mv,
            flag(h.allow_high_precision_mv),
        ),
        (frame::refreshEntropyProbs, flag(h.refresh_frame_context)),
    ]
    .into_iter()
    .fold(0, |word, (bits, value)| bits.put(word, value));
    // Its fields' 14 bits.
    v.frame_word = c_ushort::try_from(frame_word).unwrap_or(0);
    let [last_bias, golden_bias, alt_bias] = h.ref_frame_sign_bias.map(u8::from);
    v.refFrameSignBias = [0, last_bias, golden_bias, alt_bias];
    let depth_minus_8 = h.color.bit_depth.saturating_sub(8);
    (v.bitDepthMinus8Luma, v.bitDepthMinus8Chroma) = (depth_minus_8, depth_minus_8);

    let lf = &h.loop_filter;
    v.loopFilterLevel = lf.level;
    v.loopFilterSharpness = lf.sharpness;
    v.modeRefLfEnabled = u8::from(lf.delta_enabled);
    // The deltas as `int`s in `unsigned int`s, as ffmpeg assigns them.
    v.mbRefLfDelta = lf.ref_deltas.map(|delta| i32::from(delta).cast_unsigned());
    v.mbModeLfDelta = lf.mode_deltas.map(|delta| i32::from(delta).cast_unsigned());
    v.log2_tile_columns = h.tile_cols_log2;
    v.log2_tile_rows = h.tile_rows_log2;

    let seg = &h.segmentation;
    let segmentation_word = [
        (segmentation::segmentEnabled, flag(seg.enabled)),
        (segmentation::segmentMapUpdate, flag(seg.update_map)),
        (
            segmentation::segmentMapTemporalUpdate,
            flag(seg.temporal_update),
        ),
        (
            segmentation::segmentFeatureMode,
            flag(seg.abs_or_delta_update),
        ),
    ]
    .into_iter()
    .fold(0, |word, (bits, value)| bits.put(word, value));
    // Its fields' 4 bits.
    v.segmentation_word = u8::try_from(segmentation_word).unwrap_or(0);
    v.segmentFeatureEnable = seg.feature_enabled.map(|features| features.map(u8::from));
    // The quantizer, filter level and reference, and nothing for the skip
    // feature, which has no data.
    v.segmentFeatureData = seg
        .feature_data
        .map(|[q, lf, reference, _]| -> [c_short; 4] { [q, lf, reference, 0] });
    v.mb_segment_tree_probs = seg.tree_probs;
    // As the VA arm sends them: 255 where the map's prediction is not
    // temporal, where NVDEC does not read them.
    v.segment_pred_probs = if seg.temporal_update {
        seg.pred_probs
    } else {
        [255; 3]
    };

    let q = &h.quantization;
    v.qpYAc = c_int::from(q.base_q_idx);
    v.qpYDc = c_int::from(q.delta_q_y_dc);
    v.qpChDc = c_int::from(q.delta_q_uv_dc);
    v.qpChAc = c_int::from(q.delta_q_uv_ac);
    v.activeRefIdx = h.ref_frame_idx.map(c_uint::from);
    v.resetFrameContext = c_uint::from(h.reset_frame_context);
    // libvpx's numbering, the header's.
    v.mcomp_filter_type = c_uint::from(h.interp_filter);
    v.frameTagSize = c_uint::from(h.uncompressed_header_size);
    v.offsetToDctParts = c_uint::from(h.header_size_in_bytes);
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::vp9::header::{self, State};
    use crate::codec::vp9::tests::{FrameSpec, SegmentationSpec};

    /// The value of the field `bits` in `word`.
    fn field(word: u16, bits: ffi::bits::Bits) -> u32 {
        (u32::from(word) & bits.mask()) >> bits.shift
    }

    /// A key frame's fields reach `CUVIDVP9PICPARAMS` as ffmpeg's
    /// `nvdec_vp9.c` sets them, naming no reference.
    #[test]
    fn a_key_frames_header_fills_its_fields_as_ffmpeg_fills_them() {
        let mut spec = FrameSpec::key(640, 360);
        spec.filter_level = 36;
        spec.sharpness = 3;
        spec.base_q = 120;
        spec.delta_q = [-3, 2, 5];
        spec.tile_cols_increments = 1;
        let mut features = [[None; 4]; 8];
        features[2] = [Some(-20), Some(7), Some(1), Some(0)];
        spec.segmentation = Some(SegmentationSpec {
            update_map: None,
            update_data: Some((true, features)),
        });
        let bytes = spec.write();
        let (h, _) = header::parse(&bytes, &State::default(), &[None; 8]).unwrap();
        let v = vp9_parameters(&h, [NO_REFERENCE; 3]);
        assert_eq!((v.width, v.height), (640, 360));
        assert_eq!(
            [v.LastRefIdx, v.GoldenRefIdx, v.AltRefIdx],
            [NO_REFERENCE; 3]
        );
        assert_eq!(field(v.frame_word, frame::profile), 0);
        assert_eq!(field(v.frame_word, frame::frameType), 0);
        assert_eq!(field(v.frame_word, frame::showFrame), 1);
        assert_eq!(field(v.frame_word, frame::subSamplingX), 1);
        assert_eq!(field(v.frame_word, frame::subSamplingY), 1);
        assert_eq!(field(v.frame_word, frame::intraOnly), 0);
        assert_eq!(field(v.frame_word, frame::allow_high_precision_mv), 0);
        assert_eq!((v.bitDepthMinus8Luma, v.bitDepthMinus8Chroma), (0, 0));
        assert_eq!((v.loopFilterLevel, v.loopFilterSharpness), (36, 3));
        assert_eq!((v.log2_tile_columns, v.log2_tile_rows), (1, 0));
        assert_eq!((v.qpYAc, v.qpYDc, v.qpChDc, v.qpChAc), (120, -3, 2, 5));
        let seg = u16::from(v.segmentation_word);
        assert_eq!(field(seg, segmentation::segmentEnabled), 1);
        assert_eq!(field(seg, segmentation::segmentFeatureMode), 1);
        assert_eq!(v.segmentFeatureEnable[2], [1, 1, 1, 1]);
        assert_eq!(v.segmentFeatureData[2], [-20, 7, 1, 0]);
        assert_eq!(v.segmentFeatureEnable[0], [0; 4]);
        assert_eq!(v.segment_pred_probs, [255; 3]);
        // A key frame's deltas, as the specification sets them up, each an
        // `int` in an `unsigned int`.
        let minus_one = (-1i32).cast_unsigned();
        assert_eq!(v.mbRefLfDelta, [1, 0, minus_one, minus_one]);
        assert_eq!(v.mbModeLfDelta, [0, 0]);
        assert_eq!(v.frameTagSize, u32::from(h.uncompressed_header_size));
        assert_eq!(v.offsetToDctParts, u32::from(h.header_size_in_bytes));
        assert!(v.frameTagSize > 0 && v.offsetToDctParts > 0);
    }

    /// An inter frame's references, their sign biases and slots, its
    /// filter in libvpx's numbering and its 10 bits reach their fields.
    #[test]
    fn an_inter_frames_references_and_filter_reach_their_fields() {
        let mut key = FrameSpec::key(64, 64);
        key.profile = 2;
        let (_, state) = header::parse(&key.write(), &State::default(), &[None; 8]).unwrap();
        let mut spec = FrameSpec::inter(64, 64);
        spec.profile = 2;
        spec.refs = [3, 5, 7];
        spec.sign_bias = [false, true, true];
        spec.high_precision_mv = true;
        spec.context_idx = 2;
        // The coded literal 0, EIGHTTAP_SMOOTH, 1 in libvpx's numbering.
        spec.filter = Some(0);
        let (h, _) = header::parse(&spec.write(), &state, &[Some((64, 64)); 8]).unwrap();
        let v = vp9_parameters(&h, [9, 4, 0]);
        assert_eq!([v.LastRefIdx, v.GoldenRefIdx, v.AltRefIdx], [9, 4, 0]);
        assert_eq!(v.activeRefIdx, [3, 5, 7]);
        assert_eq!(v.refFrameSignBias, [0, 0, 1, 1]);
        assert_eq!(field(v.frame_word, frame::profile), 2);
        assert_eq!(field(v.frame_word, frame::frameType), 1);
        assert_eq!(field(v.frame_word, frame::frameContextIdx), 2);
        assert_eq!(field(v.frame_word, frame::allow_high_precision_mv), 1);
        assert_eq!(v.mcomp_filter_type, 1);
        assert_eq!((v.bitDepthMinus8Luma, v.bitDepthMinus8Chroma), (2, 2));
        // Nothing past the fields' bits.
        assert_eq!(v.frame_word >> 14, 0);
        assert_eq!(v.segmentation_word >> 4, 0);
    }
}
