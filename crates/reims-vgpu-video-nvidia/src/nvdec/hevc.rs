//! HEVC on NVDEC: the codec crate's decoder over an accelerator that
//! fills `CUVIDHEVCPICPARAMS` from the parsed SPS, PPS, first slice
//! segment header and reference picture set, and hands NVDEC the picture's
//! slice segments. NVIDIA's own parser is not used.
//!
//! NVDEC reads each slice segment header itself and builds the reference
//! lists from the three current sets, so those go in the standard's order
//! (8.3.2) as indices into `RefPicIdx`. The slice segments go as an
//! Annex B stream, each after a three-byte start code, with
//! `pSliceDataOffsets` giving where each starts, as for H.264; the scaling
//! lists go in coding (up-right diagonal) order, which
//! dynlink_cuviddec.h names ("scaling lists (diag order)") and the codec
//! crate keeps them in.

use std::ffi::{c_int, c_ushort};
use std::sync::Arc;

use reims_vgpu_video_codec::hevc::nal::{is_idr, is_irap};
use reims_vgpu_video_codec::hevc::sps::Sps;
use reims_vgpu_video_codec::hevc::{Accelerator, PictureParams, SliceParams};

use super::decoder::{
    Decoder, DecoderSpec, InPlace, Picture, SPARE_SURFACES, active, surface_count,
};
use super::ffi::{self, Zeroed};
use super::{ChromaFormat, NvdecCodec};
use crate::cuda::Gpu;
use crate::error::Error;

/// The NVDEC half of the HEVC decoder.
pub(crate) struct NvdecHevc {
    gpu: Arc<Gpu>,
    decoder: Option<Decoder>,
    /// The sample being decoded, for its slices to go to NVDEC in place.
    pub(crate) in_place: InPlace,
}

impl NvdecHevc {
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

/// The decoder `sps`'s pictures need, with a DPB of `dpb_size` pictures.
pub(crate) fn spec(sps: &Sps, dpb_size: usize) -> DecoderSpec {
    DecoderSpec {
        codec: NvdecCodec::Hevc,
        chroma: ChromaFormat::Yuv420,
        bit_depth: u32::from(sps.bit_depth_luma_minus8) + 8,
        width: sps.pic_width_in_luma_samples,
        height: sps.pic_height_in_luma_samples,
        surfaces: surface_count(dpb_size + 1 + SPARE_SURFACES),
    }
}

impl Accelerator for NvdecHevc {
    type Picture = Picture;
    type Error = Error;

    fn configure(&mut self, sps: &Sps, dpb_size: usize) -> Result<(), Error> {
        self.decoder = None;
        self.decoder = Some(Decoder::new(Arc::clone(&self.gpu), &spec(sps, dpb_size))?);
        Ok(())
    }

    fn new_picture(&mut self) -> Result<Picture, Error> {
        self.decoder()?.picture()
    }

    fn decode_picture(
        &mut self,
        picture: &Picture,
        params: &PictureParams<'_, Picture>,
        slices: &[SliceParams<'_>],
    ) -> Result<(), Error> {
        let mut pic = picture_parameters(picture, params)?;
        active(self.decoder.as_ref())?.decode_nals(
            &mut pic,
            slices.iter().map(|slice| slice.nal),
            &mut self.in_place,
        )
    }
}

fn narrow(value: impl TryInto<u8>) -> u8 {
    value.try_into().unwrap_or(u8::MAX)
}

fn int(value: impl TryInto<c_int>) -> c_int {
    value.try_into().unwrap_or(c_int::MAX)
}

/// `set`'s indices into the references, as `RefPicSetStCurrBefore` and
/// the others take them, and how many there are.
fn current_set(set: &[usize]) -> Result<([u8; 8], c_int), Error> {
    let mut out = [0; 8];
    if set.len() > out.len() {
        return Err(Error::Unsupported(format!(
            "{} pictures in one of the current reference sets: NVDEC takes 8",
            set.len()
        )));
    }
    for (slot, &index) in out.iter_mut().zip(set) {
        *slot = narrow(index);
    }
    Ok((out, int(set.len())))
}

/// The picture's `CUVIDPICPARAMS`, all but its bitstream.
fn picture_parameters(
    picture: &Picture,
    p: &PictureParams<'_, Picture>,
) -> Result<ffi::CUVIDPICPARAMS, Error> {
    let (sps, pps, header) = (p.sps, p.pps, p.header);
    let mut out = ffi::CUVIDPICPARAMS::zeroed();
    // In 16-pixel units, as ffmpeg's nvdec_hevc.c gives them.
    out.PicWidthInMbs = int(sps.pic_width_in_luma_samples / 16);
    out.FrameHeightInMbs = int(sps.pic_height_in_luma_samples / 16);
    out.CurrPicIdx = picture.index();
    out.ref_pic_flag = 1;
    out.intra_pic_flag = c_int::from(p.intra_only);

    let mut h = ffi::CUVIDHEVCPICPARAMS::zeroed();
    let flag = u8::from;
    let pcm = sps.pcm.unwrap_or_default();
    h.pic_width_in_luma_samples = int(sps.pic_width_in_luma_samples);
    h.pic_height_in_luma_samples = int(sps.pic_height_in_luma_samples);
    h.log2_min_luma_coding_block_size_minus3 = sps.log2_min_luma_coding_block_size_minus3;
    h.log2_diff_max_min_luma_coding_block_size = sps.log2_diff_max_min_luma_coding_block_size;
    h.log2_min_transform_block_size_minus2 = sps.log2_min_luma_transform_block_size_minus2;
    h.log2_diff_max_min_transform_block_size = sps.log2_diff_max_min_luma_transform_block_size;
    h.pcm_enabled_flag = flag(sps.pcm.is_some());
    h.log2_min_pcm_luma_coding_block_size_minus3 = pcm.log2_min_luma_coding_block_size_minus3;
    h.log2_diff_max_min_pcm_luma_coding_block_size = pcm.log2_diff_max_min_luma_coding_block_size;
    h.pcm_sample_bit_depth_luma_minus1 = pcm.sample_bit_depth_luma_minus1;
    h.pcm_sample_bit_depth_chroma_minus1 = pcm.sample_bit_depth_chroma_minus1;
    h.pcm_loop_filter_disabled_flag = flag(pcm.loop_filter_disabled_flag);
    h.strong_intra_smoothing_enabled_flag = flag(sps.strong_intra_smoothing_enabled_flag);
    h.max_transform_hierarchy_depth_intra = sps.max_transform_hierarchy_depth_intra;
    h.max_transform_hierarchy_depth_inter = sps.max_transform_hierarchy_depth_inter;
    h.amp_enabled_flag = flag(sps.amp_enabled_flag);
    h.separate_colour_plane_flag = flag(sps.separate_colour_plane_flag);
    h.log2_max_pic_order_cnt_lsb_minus4 = sps.log2_max_pic_order_cnt_lsb_minus4;
    h.num_short_term_ref_pic_sets = narrow(sps.short_term_ref_pic_sets.len());
    h.long_term_ref_pics_present_flag = flag(sps.long_term_ref_pics_present_flag);
    h.num_long_term_ref_pics_sps = narrow(sps.long_term_ref_pics.len());
    h.sps_temporal_mvp_enabled_flag = flag(sps.sps_temporal_mvp_enabled_flag);
    h.sample_adaptive_offset_enabled_flag = flag(sps.sample_adaptive_offset_enabled_flag);
    h.scaling_list_enable_flag = flag(sps.scaling_list_enabled_flag);
    h.IrapPicFlag = flag(is_irap(header.nal_unit_type));
    h.IdrPicFlag = flag(is_idr(header.nal_unit_type));
    h.bit_depth_luma_minus8 = sps.bit_depth_luma_minus8;
    h.bit_depth_chroma_minus8 = sps.bit_depth_chroma_minus8;

    h.dependent_slice_segments_enabled_flag = flag(pps.dependent_slice_segments_enabled_flag);
    h.slice_segment_header_extension_present_flag =
        flag(pps.slice_segment_header_extension_present_flag);
    h.sign_data_hiding_enabled_flag = flag(pps.sign_data_hiding_enabled_flag);
    h.cu_qp_delta_enabled_flag = flag(pps.cu_qp_delta_enabled_flag);
    h.diff_cu_qp_delta_depth = pps.diff_cu_qp_delta_depth;
    h.init_qp_minus26 = pps.init_qp_minus26;
    h.pps_cb_qp_offset = pps.pps_cb_qp_offset;
    h.pps_cr_qp_offset = pps.pps_cr_qp_offset;
    h.constrained_intra_pred_flag = flag(pps.constrained_intra_pred_flag);
    h.weighted_pred_flag = flag(pps.weighted_pred_flag);
    h.weighted_bipred_flag = flag(pps.weighted_bipred_flag);
    h.transform_skip_enabled_flag = flag(pps.transform_skip_enabled_flag);
    h.transquant_bypass_enabled_flag = flag(pps.transquant_bypass_enabled_flag);
    h.entropy_coding_sync_enabled_flag = flag(pps.entropy_coding_sync_enabled_flag);
    h.log2_parallel_merge_level_minus2 = pps.log2_parallel_merge_level_minus2;
    h.num_extra_slice_header_bits = pps.num_extra_slice_header_bits;
    h.loop_filter_across_tiles_enabled_flag = flag(pps.loop_filter_across_tiles_enabled_flag);
    h.loop_filter_across_slices_enabled_flag = flag(pps.pps_loop_filter_across_slices_enabled_flag);
    h.output_flag_present_flag = flag(pps.output_flag_present_flag);
    h.num_ref_idx_l0_default_active_minus1 = pps.num_ref_idx_l0_default_active_minus1;
    h.num_ref_idx_l1_default_active_minus1 = pps.num_ref_idx_l1_default_active_minus1;
    h.lists_modification_present_flag = flag(pps.lists_modification_present_flag);
    h.cabac_init_present_flag = flag(pps.cabac_init_present_flag);
    h.pps_slice_chroma_qp_offsets_present_flag = flag(pps.pps_slice_chroma_qp_offsets_present_flag);
    h.deblocking_filter_override_enabled_flag = flag(pps.deblocking_filter_override_enabled_flag);
    h.pps_deblocking_filter_disabled_flag = flag(pps.pps_deblocking_filter_disabled_flag);
    h.pps_beta_offset_div2 = pps.pps_beta_offset_div2;
    h.pps_tc_offset_div2 = pps.pps_tc_offset_div2;
    h.tiles_enabled_flag = flag(pps.tiles_enabled_flag);
    h.uniform_spacing_flag = flag(pps.uniform_spacing_flag);
    h.num_tile_columns_minus1 = pps.num_tile_columns_minus1;
    h.num_tile_rows_minus1 = pps.num_tile_rows_minus1;
    // Each tile's size, with uniform spacing too, as ffmpeg fills them.
    let narrow_u16 = |v: u32| c_ushort::try_from(v.saturating_sub(1)).unwrap_or(c_ushort::MAX);
    for (slot, &width) in h
        .column_width_minus1
        .iter_mut()
        .zip(&pps.column_widths(sps)?)
    {
        *slot = narrow_u16(width);
    }
    for (slot, &height) in h.row_height_minus1.iter_mut().zip(&pps.row_heights(sps)?) {
        *slot = narrow_u16(height);
    }

    h.NumBitsForShortTermRPSInSlice = int(header.st_rps_bits);
    h.NumDeltaPocsOfRefRpsIdx = int(header.st_rps_reference_delta_pocs);
    h.NumPocTotalCurr = int(header.num_pic_total_curr);
    h.CurrPicOrderCntVal = p.poc;
    h.RefPicIdx = [-1; 16];
    for (i, reference) in p.references.iter().enumerate().take(16) {
        h.RefPicIdx[i] = reference.picture.index();
        h.PicOrderCntVal[i] = reference.poc;
        h.IsLongTerm[i] = flag(reference.long_term);
    }
    let [before, after, long_term] = &p.current_sets;
    (h.RefPicSetStCurrBefore, h.NumPocStCurrBefore) = current_set(before)?;
    (h.RefPicSetStCurrAfter, h.NumPocStCurrAfter) = current_set(after)?;
    (h.RefPicSetLtCurr, h.NumPocLtCurr) = current_set(long_term)?;

    if let Some(lists) = &p.scaling_lists {
        h.ScalingList4x4 = lists.list_4x4;
        h.ScalingList8x8 = lists.list_8x8;
        h.ScalingList16x16 = lists.list_16x16;
        // 32x32 has two lists, matrixId 0 and 3 in the standard's
        // numbering (7.3.4).
        h.ScalingList32x32 = [lists.list_32x32[0], lists.list_32x32[3]];
        h.ScalingListDCCoeff16x16 = lists.dc_16x16;
        h.ScalingListDCCoeff32x32 = [lists.dc_32x32[0], lists.dc_32x32[3]];
    }
    out.CodecSpecific.hevc = h;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_current_set_holds_eight_indices_and_refuses_a_ninth() -> Result<(), Error> {
        assert_eq!(current_set(&[3, 0, 7])?, ([3, 0, 7, 0, 0, 0, 0, 0], 3));
        assert_eq!(current_set(&[])?, ([0; 8], 0));
        assert_eq!(current_set(&[1; 8])?.1, 8);
        assert!(matches!(current_set(&[1; 9]), Err(Error::Unsupported(_))));
        Ok(())
    }
}
