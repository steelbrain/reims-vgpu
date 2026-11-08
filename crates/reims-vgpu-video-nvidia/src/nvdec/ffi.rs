//! NVDEC's declarations (libnvcuvid's decoder half, `cuviddec.h`), the
//! subset this crate uses: capabilities, the decoder, picture parameters
//! for H.264 and HEVC, mapping and status. NVIDIA's parser
//! (`nvcuvid.h`) is not used: our own parsers fill the picture parameters.
//!
//! Field names are the header's, so that `crate::layout`'s test can check
//! each size, offset and bit-field against ffnvcodec's `dynlink_cuviddec.h`
//! (NVDECAPI 13.1). A C bit-field word is one Rust integer, named for what
//! it holds, with its fields' positions in the [`bits`] submodules. Those
//! words' names, and the names of the header's anonymous structs and
//! unions, are ours, each marked so.

#![allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    clippy::upper_case_acronyms,
    reason = "NVIDIA's names, kept for grepping the headers and for the layout check"
)]

use std::ffi::{c_int, c_short, c_uint, c_ulong, c_ulonglong, c_ushort, c_void};

use crate::cuda::ffi::{CUresult, CUstream};
pub(crate) use crate::dl::Zeroed;
use crate::dl::functions;

/// `CUvideodecoder`.
pub(crate) type CUvideodecoder = *mut c_void;
/// `CUvideoctxlock`.
pub(crate) type CUvideoctxlock = *mut c_void;
/// `CUvideotimestamp`.
pub(crate) type CUvideotimestamp = i64;
/// `cudaVideoCodec`.
pub(crate) type cudaVideoCodec = c_uint;
/// `cudaVideoChromaFormat`.
pub(crate) type cudaVideoChromaFormat = c_uint;
/// `cudaVideoSurfaceFormat`.
pub(crate) type cudaVideoSurfaceFormat = c_uint;
/// `cudaVideoDeinterlaceMode`.
pub(crate) type cudaVideoDeinterlaceMode = c_uint;
/// `cuvidDecodeStatus`.
pub(crate) type cuvidDecodeStatus = c_uint;

pub(crate) const cudaVideoCodec_H264: cudaVideoCodec = 4;
pub(crate) const cudaVideoCodec_JPEG: cudaVideoCodec = 5;
pub(crate) const cudaVideoCodec_HEVC: cudaVideoCodec = 8;
pub(crate) const cudaVideoCodec_VP8: cudaVideoCodec = 9;
pub(crate) const cudaVideoCodec_VP9: cudaVideoCodec = 10;
pub(crate) const cudaVideoCodec_AV1: cudaVideoCodec = 11;
pub(crate) const cudaVideoCodec_MPEG2: cudaVideoCodec = 1;

pub(crate) const cudaVideoChromaFormat_Monochrome: cudaVideoChromaFormat = 0;
pub(crate) const cudaVideoChromaFormat_420: cudaVideoChromaFormat = 1;
pub(crate) const cudaVideoChromaFormat_422: cudaVideoChromaFormat = 2;
pub(crate) const cudaVideoChromaFormat_444: cudaVideoChromaFormat = 3;

pub(crate) const cudaVideoSurfaceFormat_NV12: cudaVideoSurfaceFormat = 0;
pub(crate) const cudaVideoSurfaceFormat_P016: cudaVideoSurfaceFormat = 1;
pub(crate) const cudaVideoSurfaceFormat_YUV444: cudaVideoSurfaceFormat = 2;
pub(crate) const cudaVideoSurfaceFormat_YUV444_16Bit: cudaVideoSurfaceFormat = 3;
pub(crate) const cudaVideoSurfaceFormat_NV16: cudaVideoSurfaceFormat = 4;
pub(crate) const cudaVideoSurfaceFormat_P216: cudaVideoSurfaceFormat = 5;

pub(crate) const cudaVideoDeinterlaceMode_Weave: cudaVideoDeinterlaceMode = 0;

/// `cudaVideoCreate_PreferCUVID`: decode on the NVDEC engine, with no CUDA
/// post-processing or context lock.
pub(crate) const cudaVideoCreate_PreferCUVID: c_ulong = 0x04;

pub(crate) const cuvidDecodeStatus_InProgress: cuvidDecodeStatus = 1;
pub(crate) const cuvidDecodeStatus_Error: cuvidDecodeStatus = 8;
pub(crate) const cuvidDecodeStatus_Error_Concealed: cuvidDecodeStatus = 9;

/// `CUVIDDECODECAPS`: what NVDEC decodes of one codec, chroma format and
/// bit depth. The first three fields are in, the rest out.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CUVIDDECODECAPS {
    pub(crate) eCodecType: cudaVideoCodec,
    pub(crate) eChromaFormat: cudaVideoChromaFormat,
    pub(crate) nBitDepthMinus8: c_uint,
    pub(crate) reserved1: [c_uint; 3],
    pub(crate) bIsSupported: u8,
    pub(crate) nNumNVDECs: u8,
    pub(crate) nOutputFormatMask: c_ushort,
    pub(crate) nMaxWidth: c_uint,
    pub(crate) nMaxHeight: c_uint,
    pub(crate) nMaxMBCount: c_uint,
    pub(crate) nMinWidth: c_ushort,
    pub(crate) nMinHeight: c_ushort,
    pub(crate) bIsHistogramSupported: u8,
    pub(crate) nCounterBitDepth: u8,
    pub(crate) nMaxHistogramBins: c_ushort,
    pub(crate) bIsDecodeStatsSupported: u8,
    pub(crate) reserved4: [u8; 3],
    pub(crate) reserved3: [c_uint; 9],
}

/// A rectangle of `CUVIDDECODECREATEINFO`. Name is ours: the header's
/// `display_area` and `target_rect` are anonymous structs.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CuvidRect {
    pub(crate) left: c_short,
    pub(crate) top: c_short,
    pub(crate) right: c_short,
    pub(crate) bottom: c_short,
}

/// `CUVIDDECODECREATEINFO`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CUVIDDECODECREATEINFO {
    pub(crate) ulWidth: c_ulong,
    pub(crate) ulHeight: c_ulong,
    pub(crate) ulNumDecodeSurfaces: c_ulong,
    pub(crate) CodecType: cudaVideoCodec,
    pub(crate) ChromaFormat: cudaVideoChromaFormat,
    pub(crate) ulCreationFlags: c_ulong,
    pub(crate) bitDepthMinus8: c_ulong,
    pub(crate) ulIntraDecodeOnly: c_ulong,
    pub(crate) ulMaxWidth: c_ulong,
    pub(crate) ulMaxHeight: c_ulong,
    pub(crate) Reserved1: c_ulong,
    pub(crate) display_area: CuvidRect,
    pub(crate) OutputFormat: cudaVideoSurfaceFormat,
    pub(crate) DeinterlaceMode: cudaVideoDeinterlaceMode,
    pub(crate) ulTargetWidth: c_ulong,
    pub(crate) ulTargetHeight: c_ulong,
    pub(crate) ulNumOutputSurfaces: c_ulong,
    pub(crate) vidLock: CUvideoctxlock,
    pub(crate) target_rect: CuvidRect,
    pub(crate) enableHistogram: c_ulong,
    pub(crate) enableDecodeFeatures: c_ulong,
    pub(crate) Reserved2: [c_ulong; 3],
}

/// `CUVIDH264DPBENTRY`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CUVIDH264DPBENTRY {
    pub(crate) PicIdx: c_int,
    pub(crate) FrameIdx: c_int,
    pub(crate) is_long_term: c_int,
    pub(crate) not_existing: c_int,
    pub(crate) used_for_reference: c_int,
    pub(crate) FieldOrderCnt: [c_int; 2],
}

/// `CUVIDH264PICPARAMS`. Its last member, an anonymous union of the MVC
/// and SVC extensions, is held as its bytes: neither is used.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CUVIDH264PICPARAMS {
    pub(crate) log2_max_frame_num_minus4: c_int,
    pub(crate) pic_order_cnt_type: c_int,
    pub(crate) log2_max_pic_order_cnt_lsb_minus4: c_int,
    pub(crate) delta_pic_order_always_zero_flag: c_int,
    pub(crate) frame_mbs_only_flag: c_int,
    pub(crate) direct_8x8_inference_flag: c_int,
    pub(crate) num_ref_frames: c_int,
    pub(crate) residual_colour_transform_flag: u8,
    pub(crate) bit_depth_luma_minus8: u8,
    pub(crate) bit_depth_chroma_minus8: u8,
    pub(crate) qpprime_y_zero_transform_bypass_flag: u8,
    pub(crate) entropy_coding_mode_flag: c_int,
    pub(crate) pic_order_present_flag: c_int,
    pub(crate) num_ref_idx_l0_active_minus1: c_int,
    pub(crate) num_ref_idx_l1_active_minus1: c_int,
    pub(crate) weighted_pred_flag: c_int,
    pub(crate) weighted_bipred_idc: c_int,
    pub(crate) pic_init_qp_minus26: c_int,
    pub(crate) deblocking_filter_control_present_flag: c_int,
    pub(crate) redundant_pic_cnt_present_flag: c_int,
    pub(crate) transform_8x8_mode_flag: c_int,
    pub(crate) MbaffFrameFlag: c_int,
    pub(crate) constrained_intra_pred_flag: c_int,
    pub(crate) chroma_qp_index_offset: c_int,
    pub(crate) second_chroma_qp_index_offset: c_int,
    pub(crate) ref_pic_flag: c_int,
    pub(crate) frame_num: c_int,
    pub(crate) CurrFieldOrderCnt: [c_int; 2],
    pub(crate) dpb: [CUVIDH264DPBENTRY; 16],
    pub(crate) WeightScale4x4: [[u8; 16]; 6],
    pub(crate) WeightScale8x8: [[u8; 64]; 2],
    pub(crate) fmo_aso_enable: u8,
    pub(crate) num_slice_groups_minus1: u8,
    pub(crate) slice_group_map_type: u8,
    pub(crate) pic_init_qs_minus26: i8,
    pub(crate) slice_group_change_rate_minus1: c_uint,
    /// A union of `slice_group_map_addr` and `pMb2SliceGroupMap`.
    pub(crate) fmo: c_ulonglong,
    /// `mb_adaptive_frame_field_flag : 2` and `Reserved1 : 30`: [`bits::h264`].
    /// Name is ours.
    pub(crate) mbaff_word: c_uint,
    pub(crate) Reserved: [c_uint; 11],
    /// The anonymous union of `mvcext` and `svcext`. Name is ours: its
    /// first member's.
    pub(crate) mvcext: [u64; 18],
}

/// `CUVIDHEVCPICPARAMS`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CUVIDHEVCPICPARAMS {
    pub(crate) pic_width_in_luma_samples: c_int,
    pub(crate) pic_height_in_luma_samples: c_int,
    pub(crate) log2_min_luma_coding_block_size_minus3: u8,
    pub(crate) log2_diff_max_min_luma_coding_block_size: u8,
    pub(crate) log2_min_transform_block_size_minus2: u8,
    pub(crate) log2_diff_max_min_transform_block_size: u8,
    pub(crate) pcm_enabled_flag: u8,
    pub(crate) log2_min_pcm_luma_coding_block_size_minus3: u8,
    pub(crate) log2_diff_max_min_pcm_luma_coding_block_size: u8,
    pub(crate) pcm_sample_bit_depth_luma_minus1: u8,
    pub(crate) pcm_sample_bit_depth_chroma_minus1: u8,
    pub(crate) pcm_loop_filter_disabled_flag: u8,
    pub(crate) strong_intra_smoothing_enabled_flag: u8,
    pub(crate) max_transform_hierarchy_depth_intra: u8,
    pub(crate) max_transform_hierarchy_depth_inter: u8,
    pub(crate) amp_enabled_flag: u8,
    pub(crate) separate_colour_plane_flag: u8,
    pub(crate) log2_max_pic_order_cnt_lsb_minus4: u8,
    pub(crate) num_short_term_ref_pic_sets: u8,
    pub(crate) long_term_ref_pics_present_flag: u8,
    pub(crate) num_long_term_ref_pics_sps: u8,
    pub(crate) sps_temporal_mvp_enabled_flag: u8,
    pub(crate) sample_adaptive_offset_enabled_flag: u8,
    pub(crate) scaling_list_enable_flag: u8,
    pub(crate) IrapPicFlag: u8,
    pub(crate) IdrPicFlag: u8,
    pub(crate) bit_depth_luma_minus8: u8,
    pub(crate) bit_depth_chroma_minus8: u8,
    pub(crate) log2_max_transform_skip_block_size_minus2: u8,
    pub(crate) log2_sao_offset_scale_luma: u8,
    pub(crate) log2_sao_offset_scale_chroma: u8,
    pub(crate) high_precision_offsets_enabled_flag: u8,
    pub(crate) mv_hevc_enable: u8,
    pub(crate) nuh_layer_id: u8,
    pub(crate) default_ref_layers_active_flag: u8,
    pub(crate) NumDirectRefLayers: u8,
    pub(crate) max_one_active_ref_layer_flag: u8,
    pub(crate) poc_lsb_not_present_flag: u8,
    pub(crate) NumActiveRefLayerPics0: u8,
    pub(crate) NumActiveRefLayerPics1: u8,
    pub(crate) reserved1: [u8; 2],
    pub(crate) dependent_slice_segments_enabled_flag: u8,
    pub(crate) slice_segment_header_extension_present_flag: u8,
    pub(crate) sign_data_hiding_enabled_flag: u8,
    pub(crate) cu_qp_delta_enabled_flag: u8,
    pub(crate) diff_cu_qp_delta_depth: u8,
    pub(crate) init_qp_minus26: i8,
    pub(crate) pps_cb_qp_offset: i8,
    pub(crate) pps_cr_qp_offset: i8,
    pub(crate) constrained_intra_pred_flag: u8,
    pub(crate) weighted_pred_flag: u8,
    pub(crate) weighted_bipred_flag: u8,
    pub(crate) transform_skip_enabled_flag: u8,
    pub(crate) transquant_bypass_enabled_flag: u8,
    pub(crate) entropy_coding_sync_enabled_flag: u8,
    pub(crate) log2_parallel_merge_level_minus2: u8,
    pub(crate) num_extra_slice_header_bits: u8,
    pub(crate) loop_filter_across_tiles_enabled_flag: u8,
    pub(crate) loop_filter_across_slices_enabled_flag: u8,
    pub(crate) output_flag_present_flag: u8,
    pub(crate) num_ref_idx_l0_default_active_minus1: u8,
    pub(crate) num_ref_idx_l1_default_active_minus1: u8,
    pub(crate) lists_modification_present_flag: u8,
    pub(crate) cabac_init_present_flag: u8,
    pub(crate) pps_slice_chroma_qp_offsets_present_flag: u8,
    pub(crate) deblocking_filter_override_enabled_flag: u8,
    pub(crate) pps_deblocking_filter_disabled_flag: u8,
    pub(crate) pps_beta_offset_div2: i8,
    pub(crate) pps_tc_offset_div2: i8,
    pub(crate) tiles_enabled_flag: u8,
    pub(crate) uniform_spacing_flag: u8,
    pub(crate) num_tile_columns_minus1: u8,
    pub(crate) num_tile_rows_minus1: u8,
    pub(crate) column_width_minus1: [c_ushort; 21],
    pub(crate) row_height_minus1: [c_ushort; 21],
    pub(crate) sps_range_extension_flag: u8,
    pub(crate) transform_skip_rotation_enabled_flag: u8,
    pub(crate) transform_skip_context_enabled_flag: u8,
    pub(crate) implicit_rdpcm_enabled_flag: u8,
    pub(crate) explicit_rdpcm_enabled_flag: u8,
    pub(crate) extended_precision_processing_flag: u8,
    pub(crate) intra_smoothing_disabled_flag: u8,
    pub(crate) persistent_rice_adaptation_enabled_flag: u8,
    pub(crate) cabac_bypass_alignment_enabled_flag: u8,
    pub(crate) pps_range_extension_flag: u8,
    pub(crate) cross_component_prediction_enabled_flag: u8,
    pub(crate) chroma_qp_offset_list_enabled_flag: u8,
    pub(crate) diff_cu_chroma_qp_offset_depth: u8,
    pub(crate) chroma_qp_offset_list_len_minus1: u8,
    pub(crate) cb_qp_offset_list: [i8; 6],
    pub(crate) cr_qp_offset_list: [i8; 6],
    pub(crate) reserved2: [u8; 2],
    pub(crate) reserved3: [c_uint; 8],
    pub(crate) NumBitsForShortTermRPSInSlice: c_int,
    pub(crate) NumDeltaPocsOfRefRpsIdx: c_int,
    pub(crate) NumPocTotalCurr: c_int,
    pub(crate) NumPocStCurrBefore: c_int,
    pub(crate) NumPocStCurrAfter: c_int,
    pub(crate) NumPocLtCurr: c_int,
    pub(crate) CurrPicOrderCntVal: c_int,
    pub(crate) RefPicIdx: [c_int; 16],
    pub(crate) PicOrderCntVal: [c_int; 16],
    pub(crate) IsLongTerm: [u8; 16],
    pub(crate) RefPicSetStCurrBefore: [u8; 8],
    pub(crate) RefPicSetStCurrAfter: [u8; 8],
    pub(crate) RefPicSetLtCurr: [u8; 8],
    pub(crate) RefPicSetInterLayer0: [u8; 8],
    pub(crate) RefPicSetInterLayer1: [u8; 8],
    pub(crate) reserved4: [c_uint; 12],
    pub(crate) ScalingList4x4: [[u8; 16]; 6],
    pub(crate) ScalingList8x8: [[u8; 64]; 6],
    pub(crate) ScalingList16x16: [[u8; 64]; 6],
    pub(crate) ScalingList32x32: [[u8; 64]; 2],
    pub(crate) ScalingListDCCoeff16x16: [u8; 6],
    pub(crate) ScalingListDCCoeff32x32: [u8; 2],
}

/// `CUVIDPICPARAMS::CodecSpecific`. Name is ours: an anonymous union in the
/// header.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) union CuvidCodecSpecific {
    pub(crate) h264: CUVIDH264PICPARAMS,
    pub(crate) hevc: CUVIDHEVCPICPARAMS,
    pub(crate) CodecReserved: [c_uint; 1024],
}

/// `CUVIDPICPARAMS`: one picture to decode.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CUVIDPICPARAMS {
    pub(crate) PicWidthInMbs: c_int,
    pub(crate) FrameHeightInMbs: c_int,
    pub(crate) CurrPicIdx: c_int,
    pub(crate) field_pic_flag: c_int,
    pub(crate) bottom_field_flag: c_int,
    pub(crate) second_field: c_int,
    pub(crate) nBitstreamDataLen: c_uint,
    pub(crate) pBitstreamData: *const u8,
    pub(crate) nNumSlices: c_uint,
    pub(crate) pSliceDataOffsets: *const c_uint,
    pub(crate) ref_pic_flag: c_int,
    pub(crate) intra_pic_flag: c_int,
    pub(crate) timestamp: CUvideotimestamp,
    pub(crate) Reserved: [c_uint; 28],
    pub(crate) CodecSpecific: CuvidCodecSpecific,
}

/// `CUVIDPROCPARAMS`: how to map a decoded picture.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CUVIDPROCPARAMS {
    pub(crate) progressive_frame: c_int,
    pub(crate) second_field: c_int,
    pub(crate) top_field_first: c_int,
    pub(crate) unpaired_field: c_int,
    pub(crate) reserved_flags: c_uint,
    pub(crate) reserved_zero: c_uint,
    pub(crate) raw_input_dptr: c_ulonglong,
    pub(crate) raw_input_pitch: c_uint,
    pub(crate) raw_input_format: c_uint,
    pub(crate) raw_output_dptr: c_ulonglong,
    pub(crate) raw_output_pitch: c_uint,
    pub(crate) Reserved1: c_uint,
    pub(crate) output_stream: CUstream,
    pub(crate) Reserved: [c_uint; 46],
    pub(crate) histogram_dptr: *mut c_ulonglong,
    pub(crate) pCuvidProcExt: *mut c_void,
}

/// `CUVIDGETDECODESTATUS`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CUVIDGETDECODESTATUS {
    pub(crate) decodeStatus: cuvidDecodeStatus,
    pub(crate) reserved: [c_uint; 31],
    pub(crate) pReserved: [*mut c_void; 8],
}

/// `CUVIDRECONFIGUREDECODERINFO`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CUVIDRECONFIGUREDECODERINFO {
    pub(crate) ulWidth: c_uint,
    pub(crate) ulHeight: c_uint,
    pub(crate) ulTargetWidth: c_uint,
    pub(crate) ulTargetHeight: c_uint,
    pub(crate) ulNumDecodeSurfaces: c_uint,
    pub(crate) reserved1: [c_uint; 12],
    pub(crate) display_area: CuvidRect,
    pub(crate) target_rect: CuvidRect,
    pub(crate) reserved2: [c_uint; 11],
}

/// The positions of the C bit-fields, in the Rust words that hold them:
/// GCC's x86-64 layout, each field from the least significant bit up in
/// declaration order (checked by `crate::layout`'s test).
pub(crate) mod bits {
    /// A bit-field: `width` bits from bit `shift` of its word.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) struct Bits {
        pub(crate) shift: u32,
        pub(crate) width: u32,
    }

    impl Bits {
        /// The field's mask in its word.
        pub(crate) const fn mask(self) -> u32 {
            (u32::MAX >> (32 - self.width)) << self.shift
        }

        /// `word` with the field set to `value`, cut to the field's width.
        pub(crate) const fn put(self, word: u32, value: u32) -> u32 {
            (word & !self.mask()) | ((value << self.shift) & self.mask())
        }
    }

    const fn b(shift: u32, width: u32) -> Bits {
        Bits { shift, width }
    }

    /// `CUVIDH264PICPARAMS::mb_adaptive_frame_field_flag`.
    pub(crate) mod h264 {
        use super::{Bits, b};
        pub(crate) const mb_adaptive_frame_field_flag: Bits = b(0, 2);
    }
}

functions! {
    /// libnvcuvid's decoder functions.
    pub(crate) struct NvdecFunctions {
        cuvidGetDecoderCaps: fn(*mut CUVIDDECODECAPS) -> CUresult = c"cuvidGetDecoderCaps";
        cuvidCreateDecoder: fn(*mut CUvideodecoder, *mut CUVIDDECODECREATEINFO) -> CUresult
            = c"cuvidCreateDecoder";
        cuvidDestroyDecoder: fn(CUvideodecoder) -> CUresult = c"cuvidDestroyDecoder";
        cuvidDecodePicture: fn(CUvideodecoder, *mut CUVIDPICPARAMS) -> CUresult
            = c"cuvidDecodePicture";
        cuvidGetDecodeStatus: fn(CUvideodecoder, c_int, *mut CUVIDGETDECODESTATUS) -> CUresult
            = c"cuvidGetDecodeStatus";
        cuvidMapVideoFrame64: fn(
            CUvideodecoder,
            c_int,
            *mut c_ulonglong,
            *mut c_uint,
            *mut CUVIDPROCPARAMS,
        ) -> CUresult = c"cuvidMapVideoFrame64";
        cuvidUnmapVideoFrame64: fn(CUvideodecoder, c_ulonglong) -> CUresult
            = c"cuvidUnmapVideoFrame64";
        cuvidReconfigureDecoder: fn(CUvideodecoder, *mut CUVIDRECONFIGUREDECODERINFO) -> CUresult
            = c"cuvidReconfigureDecoder";
    }
}

// SAFETY: each is repr(C) and holds only integers, raw pointers, and
// arrays, structs and unions of those (`Zeroed`'s contract).
unsafe impl Zeroed for CUVIDDECODECAPS {}
// SAFETY: as above.
unsafe impl Zeroed for CUVIDDECODECREATEINFO {}
// SAFETY: as above.
unsafe impl Zeroed for CUVIDPICPARAMS {}
// SAFETY: as above.
unsafe impl Zeroed for CUVIDH264PICPARAMS {}
// SAFETY: as above.
unsafe impl Zeroed for CUVIDHEVCPICPARAMS {}
// SAFETY: as above.
unsafe impl Zeroed for CUVIDPROCPARAMS {}
// SAFETY: as above.
unsafe impl Zeroed for CUVIDGETDECODESTATUS {}
// SAFETY: as above.
unsafe impl Zeroed for CUVIDRECONFIGUREDECODERINFO {}

// The sizes ffnvcodec 13.1.15.0's header gives on x86-64 Linux
// (2026-10-24); `crate::layout`'s test checks every offset and bit-field
// too.
#[cfg(target_pointer_width = "64")]
const _: () = {
    assert!(size_of::<CUVIDDECODECAPS>() == 88);
    assert!(size_of::<CUVIDDECODECREATEINFO>() == 176);
    assert!(size_of::<CUVIDH264PICPARAMS>() == 984);
    assert!(size_of::<CUVIDHEVCPICPARAMS>() == 1484);
    assert!(size_of::<CUVIDPICPARAMS>() == 4280);
    assert!(size_of::<CUVIDPROCPARAMS>() == 264);
    assert!(size_of::<CUVIDGETDECODESTATUS>() == 192);
    assert!(size_of::<CUVIDRECONFIGUREDECODERINFO>() == 128);
};

#[cfg(test)]
mod tests {
    use super::bits::{Bits, h264};

    #[test]
    fn a_bit_field_is_set_alone_and_cut_to_its_width() {
        // Two 8-bit fields at the word's bottom.
        let low = Bits { shift: 0, width: 8 };
        let high = Bits { shift: 8, width: 8 };
        let word = high.put(low.put(0, 4), 2);
        assert_eq!(word, 0x0204);
        assert_eq!(low.put(word, 7), 0x0207);
        // A value wider than its field keeps to it.
        let flag = h264::mb_adaptive_frame_field_flag;
        assert_eq!(flag.put(!3_u32, 5), !2);
        assert_eq!(flag.mask().count_ones(), flag.width);
    }
}
