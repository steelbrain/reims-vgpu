//! H.264 on NVDEC: the codec crate's decoder over an accelerator that
//! fills `CUVIDH264PICPARAMS` from the parsed SPS, PPS, slice headers and
//! DPB, as the VA-API crate fills libva's buffers, and hands NVDEC the
//! picture's slices. NVIDIA's own parser is not used.
//!
//! The slices go to NVDEC as an Annex B stream, each NAL unit after a
//! three-byte start code, with `pSliceDataOffsets` giving where each
//! starts, as ffmpeg's nvdec_h264.c sends them; the scaling lists go in
//! raster order, as ffmpeg keeps them.

use std::ffi::c_int;
use std::sync::Arc;

use crate::codec::h264::{
    Accelerator, PictureParams, ReferenceFrame, SliceParams, slice::SliceType, sps::Sps,
};

use super::decoder::{
    Decoder, DecoderSpec, InPlace, Picture, SPARE_SURFACES, active, surface_count,
};
use super::ffi::{self, Zeroed};
use super::{ChromaFormat, NvdecCodec};
use crate::nvidia::cuda::Gpu;
use crate::nvidia::error::Error;

/// The NVDEC half of the H.264 decoder.
pub(crate) struct NvdecH264 {
    gpu: Arc<Gpu>,
    decoder: Option<Decoder>,
    /// The sample being decoded, for its slices to go to NVDEC in place.
    pub(crate) in_place: InPlace,
}

impl NvdecH264 {
    pub(crate) fn new(gpu: Arc<Gpu>) -> Self {
        Self {
            gpu,
            decoder: None,
            in_place: InPlace::default(),
        }
    }

    fn decoder(&self) -> Result<&Decoder, Error> {
        active(self.decoder.as_ref())
    }
}

/// The decoder `sps`'s frames need, with `dpb_frames` references.
pub(crate) fn spec(sps: &Sps, dpb_frames: usize) -> DecoderSpec {
    DecoderSpec {
        codec: NvdecCodec::H264,
        chroma: ChromaFormat::Yuv420,
        bit_depth: u32::from(sps.bit_depth_luma_minus8) + 8,
        width: sps.coded_width(),
        height: sps.coded_height(),
        surfaces: surface_count(dpb_frames + 1 + SPARE_SURFACES),
    }
}

impl Accelerator for NvdecH264 {
    type Picture = Picture;
    type Error = Error;

    fn configure(&mut self, sps: &Sps, dpb_frames: usize) -> Result<(), Error> {
        self.decoder = None;
        self.decoder = Some(Decoder::new(Arc::clone(&self.gpu), &spec(sps, dpb_frames))?);
        Ok(())
    }

    fn new_picture(&mut self) -> Result<Picture, Error> {
        self.decoder()?.picture()
    }

    /// A stand-in for a lost frame: `from`'s own surface, under a picture
    /// of its own ([`Picture::stand_in`]), since nothing but a decode
    /// writes an NVDEC surface (ours: Apple's AVC decoder conceals some way
    /// of its own). The DPB names it as NVIDIA's own parser names a lost
    /// frame ([`dpb_entries`]).
    fn conceal(&mut self, from: &Picture) -> Result<Option<Picture>, Error> {
        Ok(Some(from.stand_in()))
    }

    fn decode_picture(
        &mut self,
        picture: &Picture,
        params: &PictureParams<'_, Picture>,
        slices: &[SliceParams<'_, Picture>],
    ) -> Result<(), Error> {
        let mut pic = picture_parameters(picture, params, slices);
        active(self.decoder.as_ref())?.decode_nals(
            &mut pic,
            slices.iter().map(|slice| slice.nal),
            &mut self.in_place,
        )
    }
}

fn dpb_entry(
    reference: &ReferenceFrame<'_, Picture>,
    not_existing: bool,
) -> ffi::CUVIDH264DPBENTRY {
    ffi::CUVIDH264DPBENTRY {
        PicIdx: if not_existing {
            -1
        } else {
            reference.picture.index()
        },
        FrameIdx: c_int::try_from(reference.frame_idx).unwrap_or(0),
        is_long_term: c_int::from(reference.long_term),
        not_existing: c_int::from(not_existing),
        // Frames only: both fields are references.
        used_for_reference: 3,
        FieldOrderCnt: [reference.top_poc, reference.bottom_poc],
    }
}

/// The DPB as NVDEC takes it. A stand-in for a lost frame names its
/// source's surface where no other entry does. Where one does, it is named
/// `not_existing` with `PicIdx` -1, as NVIDIA's own parser names a lost
/// frame while the frame before it is still a reference: NVDEC failed each
/// picture whose DPB named one surface twice, with status 9
/// (`cuvidDecodeStatus_Error_Concealed`).
fn dpb_entries(references: &[ReferenceFrame<'_, Picture>]) -> Vec<ffi::CUVIDH264DPBENTRY> {
    let mut named: Vec<c_int> = references
        .iter()
        .filter(|r| !r.picture.is_stand_in())
        .map(|r| r.picture.index())
        .collect();
    references
        .iter()
        .map(|reference| {
            let index = reference.picture.index();
            let not_existing = reference.picture.is_stand_in() && named.contains(&index);
            if reference.picture.is_stand_in() && !not_existing {
                named.push(index);
            }
            dpb_entry(reference, not_existing)
        })
        .collect()
}

/// The picture's `CUVIDPICPARAMS`, all but its bitstream.
fn picture_parameters(
    picture: &Picture,
    p: &PictureParams<'_, Picture>,
    slices: &[SliceParams<'_, Picture>],
) -> ffi::CUVIDPICPARAMS {
    let (sps, pps, header) = (p.sps, p.pps, p.header);
    let reference = header.nal_ref_idc != 0;
    let mut out = ffi::CUVIDPICPARAMS::zeroed();
    out.PicWidthInMbs = c_int::try_from(sps.width_in_mbs()).unwrap_or(0);
    out.FrameHeightInMbs = c_int::try_from(sps.frame_height_in_mbs()).unwrap_or(0);
    out.CurrPicIdx = picture.index();
    out.field_pic_flag = c_int::from(header.field_pic_flag);
    out.bottom_field_flag = c_int::from(header.bottom_field_flag);
    out.second_field = 0;
    out.ref_pic_flag = c_int::from(reference);
    out.intra_pic_flag = c_int::from(
        slices
            .iter()
            .all(|s| matches!(s.header.slice_type, SliceType::I | SliceType::Si)),
    );
    let mut h = ffi::CUVIDH264PICPARAMS::zeroed();
    let flag = c_int::from;
    h.log2_max_frame_num_minus4 = c_int::from(sps.log2_max_frame_num_minus4);
    h.pic_order_cnt_type = c_int::from(sps.pic_order_cnt_type);
    h.log2_max_pic_order_cnt_lsb_minus4 = c_int::from(sps.log2_max_pic_order_cnt_lsb_minus4);
    h.delta_pic_order_always_zero_flag = flag(sps.delta_pic_order_always_zero_flag);
    h.frame_mbs_only_flag = flag(sps.frame_mbs_only_flag);
    h.direct_8x8_inference_flag = flag(sps.direct_8x8_inference_flag);
    h.num_ref_frames = c_int::from(sps.max_num_ref_frames);
    h.residual_colour_transform_flag = u8::from(sps.separate_colour_plane_flag);
    h.bit_depth_luma_minus8 = sps.bit_depth_luma_minus8;
    h.bit_depth_chroma_minus8 = sps.bit_depth_chroma_minus8;
    h.qpprime_y_zero_transform_bypass_flag = u8::from(sps.qpprime_y_zero_transform_bypass_flag);
    h.entropy_coding_mode_flag = flag(pps.entropy_coding_mode_flag);
    h.pic_order_present_flag = flag(pps.bottom_field_pic_order_in_frame_present_flag);
    // The PPS's defaults, as ffmpeg gives them: each slice's own counts
    // are in its header, which NVDEC reads.
    h.num_ref_idx_l0_active_minus1 = c_int::from(pps.num_ref_idx_l0_default_active_minus1);
    h.num_ref_idx_l1_active_minus1 = c_int::from(pps.num_ref_idx_l1_default_active_minus1);
    h.weighted_pred_flag = flag(pps.weighted_pred_flag);
    h.weighted_bipred_idc = c_int::from(pps.weighted_bipred_idc);
    h.pic_init_qp_minus26 = c_int::from(pps.pic_init_qp_minus26);
    h.deblocking_filter_control_present_flag = flag(pps.deblocking_filter_control_present_flag);
    h.redundant_pic_cnt_present_flag = flag(pps.redundant_pic_cnt_present_flag);
    h.transform_8x8_mode_flag = flag(pps.transform_8x8_mode_flag);
    h.MbaffFrameFlag = flag(sps.mb_adaptive_frame_field_flag && !header.field_pic_flag);
    h.constrained_intra_pred_flag = flag(pps.constrained_intra_pred_flag);
    h.chroma_qp_index_offset = c_int::from(pps.chroma_qp_index_offset);
    h.second_chroma_qp_index_offset = c_int::from(pps.second_chroma_qp_index_offset);
    h.ref_pic_flag = c_int::from(reference);
    h.frame_num = c_int::try_from(header.frame_num).unwrap_or(0);
    h.CurrFieldOrderCnt = [p.top_poc, p.bottom_poc];
    for entry in &mut h.dpb {
        entry.PicIdx = -1;
    }
    for (slot, entry) in h.dpb.iter_mut().zip(dpb_entries(&p.references)) {
        *slot = entry;
    }
    let lists = &p.scaling_lists;
    h.WeightScale4x4 = core::array::from_fn(|i| lists.raster_4x4(i));
    h.WeightScale8x8 = [lists.raster_8x8(0), lists.raster_8x8(1)];
    h.num_slice_groups_minus1 = pps.num_slice_groups_minus1;
    h.pic_init_qs_minus26 = pps.pic_init_qs_minus26;
    h.mbaff_word = ffi::bits::h264::mb_adaptive_frame_field_flag
        .put(0, u32::from(sps.mb_adaptive_frame_field_flag));
    out.CodecSpecific.h264 = h;
    out
}
