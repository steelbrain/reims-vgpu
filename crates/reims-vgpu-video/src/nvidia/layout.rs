//! The binding checked against NVIDIA's headers, as ffnvcodec packages
//! them (`/usr/include/ffnvcodec`): every struct's size and its fields'
//! offsets, every bit-field's position, every constant's value, every GUID,
//! and every function's signature.
//!
//! The test writes one C program from this module's tables, compiles it
//! against the installed headers and runs it; the program prints what C
//! sees and the test compares that with what Rust declares, line for line.
//! It is ignored by default, since it needs the headers and a C compiler:
//! run it on the NVIDIA test host.

use std::fmt::Write as _;
use std::io::Write as _;
use std::mem::{offset_of, size_of};
use std::process::{Command, Stdio};

use crate::nvidia::cuda::ffi as cu;
use crate::nvidia::dl::Signature;
use crate::nvidia::nvdec::ffi as dec;
use crate::nvidia::nvdec::ffi::bits::Bits;
use crate::nvidia::nvenc::ffi as enc;

/// One line each side prints, and the C that prints it.
struct Check {
    /// What Rust expects the C program to print.
    expected: String,
    /// The C statement that prints it.
    c: String,
}

#[derive(Default)]
struct Checks(Vec<Check>);

impl Checks {
    fn size(&mut self, c_type: &str, size: usize) {
        self.0.push(Check {
            expected: format!("{c_type} size {size}"),
            c: format!("printf(\"%s size %zu\\n\", {c_type:?}, sizeof({c_type}));"),
        });
    }

    fn offset(&mut self, c_type: &str, field: &str, offset: usize) {
        self.0.push(Check {
            expected: format!("{c_type}.{field} {offset}"),
            c: format!("printf(\"%s %zu\\n\", \"{c_type}.{field}\", offsetof({c_type}, {field}));"),
        });
    }

    /// A bit-field `path` (a member access from a `c_type` value) that
    /// Rust holds as `bits` of the word at `word_offset`, `word_size`
    /// bytes long: the C program sets it to all ones in a zeroed value and
    /// prints the bytes that changed.
    fn bits(&mut self, c_type: &str, path: &str, word_offset: usize, word_size: usize, bits: Bits) {
        let mask = u64::from(bits.mask()).to_le_bytes();
        let mut expected = format!("{c_type}.{path} bits");
        for (index, byte) in mask.iter().take(word_size).enumerate() {
            if *byte != 0 {
                let _ = write!(expected, " {}={byte:02x}", word_offset + index);
            }
        }
        self.0.push(Check {
            expected,
            c: format!(
                "{{ {c_type} v; memset(&v, 0, sizeof v); v.{path} = ~0u; \
                 dump(\"{c_type}.{path} bits\", (const unsigned char *)&v, sizeof v); }}"
            ),
        });
    }

    fn constant(&mut self, name: &str, value: i64) {
        self.0.push(Check {
            expected: format!("{name} {value}"),
            c: format!("printf(\"%s %lld\\n\", {name:?}, (long long)({name}));"),
        });
    }

    fn guid(&mut self, name: &str, guid: enc::GUID) {
        let mut expected = format!(
            "{name} {:08x}-{:04x}-{:04x}",
            guid.Data1, guid.Data2, guid.Data3
        );
        for byte in guid.Data4 {
            let _ = write!(expected, "-{byte:02x}");
        }
        self.0.push(Check {
            expected,
            c: format!(
                "{{ GUID g = {name}; printf(\"%s %08x-%04x-%04x\", {name:?}, g.Data1, g.Data2, \
                 g.Data3); for (int i = 0; i < 8; i++) printf(\"-%02x\", g.Data4[i]); \
                 printf(\"\\n\"); }}"
            ),
        });
    }

    /// A function whose C type is `c_type` (a function type, or a pointer
    /// to one when `pointer`), declared in Rust as `signature`.
    fn signature(&mut self, c_type: &str, signature: &Signature, pointer: bool) {
        let arguments: Vec<String> = signature.arguments.iter().map(|a| c_of(a)).collect();
        let arguments = if arguments.is_empty() {
            "void".to_owned()
        } else {
            arguments.join(", ")
        };
        let result = c_of(signature.result);
        let declared = if pointer {
            format!("{result} (*)({arguments})")
        } else {
            format!("{result} ({arguments})")
        };
        self.0.push(Check {
            expected: format!("signature {} 1", signature.symbol),
            c: format!(
                "printf(\"signature %s %d\\n\", {:?}, \
                 __builtin_types_compatible_p(__typeof__({c_type}), {declared}));",
                signature.symbol
            ),
        });
    }
}

/// The C spelling of a Rust type as `stringify!` gives it: `*mut T` is
/// `T *`, `*const T` is `T const *`, and the `std::ffi` and fixed-width
/// integers are their C names; the binding's aliases are the header's
/// names, but for `Encoder`, which is NVENC's `void *`.
fn c_of(rust: &str) -> String {
    let rust = rust.trim();
    if let Some(rest) = rust.strip_prefix('*') {
        let rest = rest.trim_start();
        if let Some(inner) = rest.strip_prefix("mut") {
            return format!("{} *", c_of(inner));
        }
        if let Some(inner) = rest.strip_prefix("const") {
            return format!("{} const *", c_of(inner));
        }
    }
    match rust {
        "c_int" => "int",
        "c_uint" => "unsigned int",
        "c_char" => "char",
        "c_void" => "void",
        "c_ulonglong" => "unsigned long long",
        "u32" => "uint32_t",
        "u64" => "uint64_t",
        "usize" => "size_t",
        "Encoder" => "void *",
        "()" => "void",
        other => other,
    }
    .to_owned()
}

macro_rules! layout {
    ($checks:ident, $ty:ty, $c:literal { $($field:ident),* $(,)? }) => {
        $checks.size($c, size_of::<$ty>());
        $( $checks.offset($c, stringify!($field), offset_of!($ty, $field)); )*
    };
}

/// The bit-fields of one C struct held in one Rust word.
macro_rules! bit_fields {
    ($checks:ident, $ty:ty, $c:expr, $word:ident: $word_ty:ty, $module:path { $($field:ident),* $(,)? }) => {{
        use $module as m;
        $(
            $checks.bits(
                $c,
                stringify!($field),
                offset_of!($ty, $word),
                size_of::<$word_ty>(),
                m::$field,
            );
        )*
    }};
}

fn cuda_checks(checks: &mut Checks) {
    layout!(checks, cu::CUDA_MEMCPY2D, "CUDA_MEMCPY2D" {
        srcXInBytes, srcY, srcMemoryType, srcHost, srcDevice, srcArray, srcPitch, dstXInBytes,
        dstY, dstMemoryType, dstHost, dstDevice, dstArray, dstPitch, WidthInBytes, Height,
    });
    checks.size("CUresult", size_of::<cu::CUresult>());
    checks.size("CUdevice", size_of::<cu::CUdevice>());
    checks.size("CUdeviceptr", size_of::<cu::CUdeviceptr>());
    checks.constant("CUDA_SUCCESS", cu::CUDA_SUCCESS.into());
    checks.constant("CU_MEMORYTYPE_HOST", cu::CU_MEMORYTYPE_HOST.into());
    checks.constant("CU_MEMORYTYPE_DEVICE", cu::CU_MEMORYTYPE_DEVICE.into());
    checks.constant("CUDA_ERROR_NOT_READY", cu::CUDA_ERROR_NOT_READY.into());
    checks.constant("CU_STREAM_NON_BLOCKING", cu::CU_STREAM_NON_BLOCKING.into());
    for (name, value) in [
        (
            "CU_DEVICE_ATTRIBUTE_PCI_BUS_ID",
            cu::CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,
        ),
        (
            "CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID",
            cu::CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,
        ),
        (
            "CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID",
            cu::CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID,
        ),
        (
            "CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR",
            cu::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        ),
        (
            "CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR",
            cu::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        ),
    ] {
        checks.constant(name, value.into());
    }
    for signature in cu::CudaFunctions::SIGNATURES {
        checks.signature(&format!("t{}", signature.symbol), signature, false);
    }
    // The handle's description: its `type` field is a Rust keyword, so checked by hand.
    let handle = "CUDA_EXTERNAL_MEMORY_HANDLE_DESC";
    checks.size(handle, size_of::<cu::CUDA_EXTERNAL_MEMORY_HANDLE_DESC>());
    checks.offset(
        handle,
        "type",
        offset_of!(cu::CUDA_EXTERNAL_MEMORY_HANDLE_DESC, r#type),
    );
    for (field, offset) in [
        (
            "handle",
            offset_of!(cu::CUDA_EXTERNAL_MEMORY_HANDLE_DESC, handle),
        ),
        (
            "size",
            offset_of!(cu::CUDA_EXTERNAL_MEMORY_HANDLE_DESC, size),
        ),
        (
            "flags",
            offset_of!(cu::CUDA_EXTERNAL_MEMORY_HANDLE_DESC, flags),
        ),
        (
            "reserved",
            offset_of!(cu::CUDA_EXTERNAL_MEMORY_HANDLE_DESC, reserved),
        ),
    ] {
        checks.offset(handle, field, offset);
    }
    checks.size(
        "__typeof__(((CUDA_EXTERNAL_MEMORY_HANDLE_DESC *)0)->handle)",
        size_of::<cu::CudaExternalMemoryHandle>(),
    );
    layout!(checks, cu::CUDA_EXTERNAL_MEMORY_BUFFER_DESC, "CUDA_EXTERNAL_MEMORY_BUFFER_DESC" {
        offset, size, flags, reserved,
    });
    checks.constant(
        "CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD",
        cu::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD.into(),
    );
    // `CUDA_EXTERNAL_MEMORY_DEDICATED` is CUDA's `cuda.h`'s, which ffnvcodec's headers do not
    // carry: not checked here.
    for signature in cu::CudaExternalMemory::SIGNATURES {
        checks.signature(&format!("t{}", signature.symbol), signature, false);
    }
}

fn nvdec_checks(checks: &mut Checks) {
    layout!(checks, dec::CUVIDDECODECAPS, "CUVIDDECODECAPS" {
        eCodecType, eChromaFormat, nBitDepthMinus8, reserved1, bIsSupported, nNumNVDECs,
        nOutputFormatMask, nMaxWidth, nMaxHeight, nMaxMBCount, nMinWidth, nMinHeight,
        bIsHistogramSupported, nCounterBitDepth, nMaxHistogramBins, bIsDecodeStatsSupported,
        reserved4, reserved3,
    });
    layout!(checks, dec::CUVIDDECODECREATEINFO, "CUVIDDECODECREATEINFO" {
        ulWidth, ulHeight, ulNumDecodeSurfaces, CodecType, ChromaFormat, ulCreationFlags,
        bitDepthMinus8, ulIntraDecodeOnly, ulMaxWidth, ulMaxHeight, Reserved1, display_area,
        OutputFormat, DeinterlaceMode, ulTargetWidth, ulTargetHeight, ulNumOutputSurfaces,
        vidLock, target_rect, enableHistogram, enableDecodeFeatures, Reserved2,
    });
    checks.size(
        "__typeof__(((CUVIDDECODECREATEINFO *)0)->display_area)",
        size_of::<dec::CuvidRect>(),
    );
    layout!(checks, dec::CUVIDH264DPBENTRY, "CUVIDH264DPBENTRY" {
        PicIdx, FrameIdx, is_long_term, not_existing, used_for_reference, FieldOrderCnt,
    });
    layout!(checks, dec::CUVIDH264PICPARAMS, "CUVIDH264PICPARAMS" {
        log2_max_frame_num_minus4, pic_order_cnt_type, log2_max_pic_order_cnt_lsb_minus4,
        delta_pic_order_always_zero_flag, frame_mbs_only_flag, direct_8x8_inference_flag,
        num_ref_frames, residual_colour_transform_flag, bit_depth_luma_minus8,
        bit_depth_chroma_minus8, qpprime_y_zero_transform_bypass_flag, entropy_coding_mode_flag,
        pic_order_present_flag, num_ref_idx_l0_active_minus1, num_ref_idx_l1_active_minus1,
        weighted_pred_flag, weighted_bipred_idc, pic_init_qp_minus26,
        deblocking_filter_control_present_flag, redundant_pic_cnt_present_flag,
        transform_8x8_mode_flag, MbaffFrameFlag, constrained_intra_pred_flag,
        chroma_qp_index_offset, second_chroma_qp_index_offset, ref_pic_flag, frame_num,
        CurrFieldOrderCnt, dpb, WeightScale4x4, WeightScale8x8, fmo_aso_enable,
        num_slice_groups_minus1, slice_group_map_type, pic_init_qs_minus26,
        slice_group_change_rate_minus1, fmo, Reserved, mvcext,
    });
    bit_fields!(checks, dec::CUVIDH264PICPARAMS, "CUVIDH264PICPARAMS", mbaff_word: u32,
        dec::bits::h264 { mb_adaptive_frame_field_flag });
    layout!(checks, dec::CUVIDHEVCPICPARAMS, "CUVIDHEVCPICPARAMS" {
        pic_width_in_luma_samples, pic_height_in_luma_samples,
        log2_min_luma_coding_block_size_minus3, log2_diff_max_min_luma_coding_block_size,
        log2_min_transform_block_size_minus2, log2_diff_max_min_transform_block_size,
        pcm_enabled_flag, log2_min_pcm_luma_coding_block_size_minus3,
        log2_diff_max_min_pcm_luma_coding_block_size, pcm_sample_bit_depth_luma_minus1,
        pcm_sample_bit_depth_chroma_minus1, pcm_loop_filter_disabled_flag,
        strong_intra_smoothing_enabled_flag, max_transform_hierarchy_depth_intra,
        max_transform_hierarchy_depth_inter, amp_enabled_flag, separate_colour_plane_flag,
        log2_max_pic_order_cnt_lsb_minus4, num_short_term_ref_pic_sets,
        long_term_ref_pics_present_flag, num_long_term_ref_pics_sps,
        sps_temporal_mvp_enabled_flag, sample_adaptive_offset_enabled_flag,
        scaling_list_enable_flag, IrapPicFlag, IdrPicFlag, bit_depth_luma_minus8,
        bit_depth_chroma_minus8, log2_max_transform_skip_block_size_minus2,
        log2_sao_offset_scale_luma, log2_sao_offset_scale_chroma,
        high_precision_offsets_enabled_flag, mv_hevc_enable, nuh_layer_id,
        default_ref_layers_active_flag, NumDirectRefLayers, max_one_active_ref_layer_flag,
        poc_lsb_not_present_flag, NumActiveRefLayerPics0, NumActiveRefLayerPics1, reserved1,
        dependent_slice_segments_enabled_flag, slice_segment_header_extension_present_flag,
        sign_data_hiding_enabled_flag, cu_qp_delta_enabled_flag, diff_cu_qp_delta_depth,
        init_qp_minus26, pps_cb_qp_offset, pps_cr_qp_offset, constrained_intra_pred_flag,
        weighted_pred_flag, weighted_bipred_flag, transform_skip_enabled_flag,
        transquant_bypass_enabled_flag, entropy_coding_sync_enabled_flag,
        log2_parallel_merge_level_minus2, num_extra_slice_header_bits,
        loop_filter_across_tiles_enabled_flag, loop_filter_across_slices_enabled_flag,
        output_flag_present_flag, num_ref_idx_l0_default_active_minus1,
        num_ref_idx_l1_default_active_minus1, lists_modification_present_flag,
        cabac_init_present_flag, pps_slice_chroma_qp_offsets_present_flag,
        deblocking_filter_override_enabled_flag, pps_deblocking_filter_disabled_flag,
        pps_beta_offset_div2, pps_tc_offset_div2, tiles_enabled_flag, uniform_spacing_flag,
        num_tile_columns_minus1, num_tile_rows_minus1, column_width_minus1, row_height_minus1,
        sps_range_extension_flag, transform_skip_rotation_enabled_flag,
        transform_skip_context_enabled_flag, implicit_rdpcm_enabled_flag,
        explicit_rdpcm_enabled_flag, extended_precision_processing_flag,
        intra_smoothing_disabled_flag, persistent_rice_adaptation_enabled_flag,
        cabac_bypass_alignment_enabled_flag, pps_range_extension_flag,
        cross_component_prediction_enabled_flag, chroma_qp_offset_list_enabled_flag,
        diff_cu_chroma_qp_offset_depth, chroma_qp_offset_list_len_minus1, cb_qp_offset_list,
        cr_qp_offset_list, reserved2, reserved3, NumBitsForShortTermRPSInSlice,
        NumDeltaPocsOfRefRpsIdx, NumPocTotalCurr, NumPocStCurrBefore, NumPocStCurrAfter,
        NumPocLtCurr, CurrPicOrderCntVal, RefPicIdx, PicOrderCntVal, IsLongTerm,
        RefPicSetStCurrBefore, RefPicSetStCurrAfter, RefPicSetLtCurr, RefPicSetInterLayer0,
        RefPicSetInterLayer1, reserved4, ScalingList4x4, ScalingList8x8, ScalingList16x16,
        ScalingList32x32, ScalingListDCCoeff16x16, ScalingListDCCoeff32x32,
    });
    layout!(checks, dec::CUVIDPICPARAMS, "CUVIDPICPARAMS" {
        PicWidthInMbs, FrameHeightInMbs, CurrPicIdx, field_pic_flag, bottom_field_flag,
        second_field, nBitstreamDataLen, pBitstreamData, nNumSlices, pSliceDataOffsets,
        ref_pic_flag, intra_pic_flag, timestamp, Reserved, CodecSpecific,
    });
    checks.size(
        "__typeof__(((CUVIDPICPARAMS *)0)->CodecSpecific)",
        size_of::<dec::CuvidCodecSpecific>(),
    );
    layout!(checks, dec::CUVIDPROCPARAMS, "CUVIDPROCPARAMS" {
        progressive_frame, second_field, top_field_first, unpaired_field, reserved_flags,
        reserved_zero, raw_input_dptr, raw_input_pitch, raw_input_format, raw_output_dptr,
        raw_output_pitch, Reserved1, output_stream, Reserved, histogram_dptr, pCuvidProcExt,
    });
    layout!(checks, dec::CUVIDGETDECODESTATUS, "CUVIDGETDECODESTATUS" {
        decodeStatus, reserved, pReserved,
    });
    layout!(checks, dec::CUVIDRECONFIGUREDECODERINFO, "CUVIDRECONFIGUREDECODERINFO" {
        ulWidth, ulHeight, ulTargetWidth, ulTargetHeight, ulNumDecodeSurfaces, reserved1,
        display_area, target_rect, reserved2,
    });
    for (name, value) in [
        ("cudaVideoCodec_H264", dec::cudaVideoCodec_H264),
        ("cudaVideoCodec_HEVC", dec::cudaVideoCodec_HEVC),
        (
            "cudaVideoChromaFormat_Monochrome",
            dec::cudaVideoChromaFormat_Monochrome,
        ),
        ("cudaVideoChromaFormat_420", dec::cudaVideoChromaFormat_420),
        ("cudaVideoChromaFormat_422", dec::cudaVideoChromaFormat_422),
        ("cudaVideoChromaFormat_444", dec::cudaVideoChromaFormat_444),
        (
            "cudaVideoSurfaceFormat_NV12",
            dec::cudaVideoSurfaceFormat_NV12,
        ),
        (
            "cudaVideoSurfaceFormat_P016",
            dec::cudaVideoSurfaceFormat_P016,
        ),
        (
            "cudaVideoSurfaceFormat_YUV444",
            dec::cudaVideoSurfaceFormat_YUV444,
        ),
        (
            "cudaVideoSurfaceFormat_YUV444_16Bit",
            dec::cudaVideoSurfaceFormat_YUV444_16Bit,
        ),
        (
            "cudaVideoSurfaceFormat_NV16",
            dec::cudaVideoSurfaceFormat_NV16,
        ),
        (
            "cudaVideoSurfaceFormat_P216",
            dec::cudaVideoSurfaceFormat_P216,
        ),
        (
            "cudaVideoDeinterlaceMode_Weave",
            dec::cudaVideoDeinterlaceMode_Weave,
        ),
        (
            "cuvidDecodeStatus_InProgress",
            dec::cuvidDecodeStatus_InProgress,
        ),
        ("cuvidDecodeStatus_Error", dec::cuvidDecodeStatus_Error),
        (
            "cuvidDecodeStatus_Error_Concealed",
            dec::cuvidDecodeStatus_Error_Concealed,
        ),
    ] {
        checks.constant(name, value.into());
    }
    checks.constant(
        "cudaVideoCreate_PreferCUVID",
        dec::cudaVideoCreate_PreferCUVID.cast_signed(),
    );
    for signature in dec::NvdecFunctions::SIGNATURES {
        checks.signature(&format!("t{}", signature.symbol), signature, false);
    }
}

fn nvenc_checks(checks: &mut Checks) {
    layout!(checks, enc::GUID, "GUID" { Data1, Data2, Data3, Data4 });
    layout!(checks, enc::NV_ENC_CAPS_PARAM, "NV_ENC_CAPS_PARAM" {
        version, capsToQuery, reserved,
    });
    layout!(checks, enc::NV_ENC_CREATE_INPUT_BUFFER, "NV_ENC_CREATE_INPUT_BUFFER" {
        version, width, height, memoryHeap, bufferFmt, reserved, inputBuffer, pSysMemBuffer,
        reserved1, reserved2,
    });
    layout!(checks, enc::NV_ENC_CREATE_BITSTREAM_BUFFER, "NV_ENC_CREATE_BITSTREAM_BUFFER" {
        version, size, memoryHeap, reserved, bitstreamBuffer, bitstreamBufferPtr, reserved1,
        reserved2,
    });
    layout!(checks, enc::NV_ENC_QP, "NV_ENC_QP" { qpInterP, qpInterB, qpIntra });
    layout!(checks, enc::NV_ENC_RC_PARAMS, "NV_ENC_RC_PARAMS" {
        version, rateControlMode, constQP, averageBitRate, maxBitRate, vbvBufferSize,
        vbvInitialDelay, minQP, maxQP, initialRCQP, temporallayerIdxMask, temporalLayerQP,
        targetQuality, targetQualityLSB, lookaheadDepth, lowDelayKeyFrameScale,
        yDcQPIndexOffset, uDcQPIndexOffset, vDcQPIndexOffset, qpMapMode, multiPass,
        alphaLayerBitrateRatio, cbQPIndexOffset, crQPIndexOffset, reserved2, lookaheadLevel,
        viewBitrateRatios, reserved3, reserved1,
    });
    bit_fields!(checks, enc::NV_ENC_RC_PARAMS, "NV_ENC_RC_PARAMS", flags: u32,
    enc::bits::rc { enableLookahead });
    layout!(checks, enc::NV_ENC_CONFIG_H264_VUI_PARAMETERS, "NV_ENC_CONFIG_H264_VUI_PARAMETERS" {
        overscanInfoPresentFlag, overscanInfo, videoSignalTypePresentFlag, videoFormat,
        videoFullRangeFlag, colourDescriptionPresentFlag, colourPrimaries,
        transferCharacteristics, colourMatrix, chromaSampleLocationFlag, chromaSampleLocationTop,
        chromaSampleLocationBot, bitstreamRestrictionFlag, timingInfoPresentFlag,
        numUnitInTicks, timeScale, reserved,
    });
    layout!(checks, enc::NV_ENC_CONFIG_H264, "NV_ENC_CONFIG_H264" {
        level, idrPeriod, separateColourPlaneFlag, disableDeblockingFilterIDC,
        numTemporalLayers, spsId, ppsId, adaptiveTransformMode, fmoMode, bdirectMode,
        entropyCodingMode, stereoMode, intraRefreshPeriod, intraRefreshCnt, maxNumRefFrames,
        sliceMode, sliceModeData, h264VUIParameters, ltrNumFrames, ltrTrustMode,
        chromaFormatIDC, maxTemporalLayers, useBFramesAsRef, numRefL0, numRefL1,
        outputBitDepth, inputBitDepth, tfLevel, reserved1, reserved2,
    });
    bit_fields!(checks, enc::NV_ENC_CONFIG_H264, "NV_ENC_CONFIG_H264", flags: u32,
    enc::bits::h264 { repeatSPSPPS });
    layout!(checks, enc::NV_ENC_CONFIG_HEVC, "NV_ENC_CONFIG_HEVC" {
        level, tier, minCUSize, maxCUSize, idrPeriod, intraRefreshPeriod, intraRefreshCnt,
        maxNumRefFramesInDPB, ltrNumFrames, vpsId, spsId, ppsId, sliceMode, sliceModeData,
        maxTemporalLayersMinus1, hevcVUIParameters, ltrTrustMode, useBFramesAsRef, numRefL0,
        numRefL1, tfLevel, disableDeblockingFilterIDC, outputBitDepth, inputBitDepth,
        numTemporalLayers, numViews, reserved1, reserved2,
    });
    bit_fields!(checks, enc::NV_ENC_CONFIG_HEVC, "NV_ENC_CONFIG_HEVC", flags: u32,
    enc::bits::hevc { repeatSPSPPS, chromaFormatIDC });
    checks.size("NV_ENC_CODEC_CONFIG", size_of::<enc::NV_ENC_CODEC_CONFIG>());
    layout!(checks, enc::NV_ENC_CONFIG, "NV_ENC_CONFIG" {
        version, profileGUID, gopLength, frameIntervalP, monoChromeEncoding, frameFieldMode,
        mvPrecision, rcParams, encodeCodecConfig, reserved, reserved2,
    });
    checks.size(
        "NVENC_EXTERNAL_ME_HINT_COUNTS_PER_BLOCKTYPE",
        size_of::<enc::NVENC_EXTERNAL_ME_HINT_COUNTS_PER_BLOCKTYPE>(),
    );
    layout!(checks, enc::NV_ENC_INITIALIZE_PARAMS, "NV_ENC_INITIALIZE_PARAMS" {
        version, encodeGUID, presetGUID, encodeWidth, encodeHeight, darWidth, darHeight,
        frameRateNum, frameRateDen, enableEncodeAsync, enablePTD, privDataSize, reserved,
        privData, encodeConfig, maxEncodeWidth, maxEncodeHeight, maxMEHintCountsPerBlock,
        tuningInfo, bufferFormat, numStateBuffers, outputStatsLevel, reserved1, reserved2,
    });
    layout!(checks, enc::NV_ENC_RECONFIGURE_PARAMS, "NV_ENC_RECONFIGURE_PARAMS" {
        version, reserved, reInitEncodeParams, reserved2,
    });
    layout!(checks, enc::NV_ENC_PRESET_CONFIG, "NV_ENC_PRESET_CONFIG" {
        version, reserved, presetCfg, reserved1, reserved2,
    });
    checks.size(
        "NV_ENC_CODEC_PIC_PARAMS",
        size_of::<enc::NV_ENC_CODEC_PIC_PARAMS>(),
    );
    layout!(checks, enc::NV_ENC_PIC_PARAMS, "NV_ENC_PIC_PARAMS" {
        version, inputWidth, inputHeight, inputPitch, encodePicFlags, frameIdx, inputTimeStamp,
        inputDuration, inputBuffer, outputBitstream, completionEvent, bufferFmt, pictureStruct,
        pictureType, codecPicParams, meHintCountsPerBlock, meExternalHints, reserved2,
        reserved5, qpDeltaMap, qpDeltaMapSize, reservedBitFields, meHintRefPicDist,
        diffPicNumHint, alphaBuffer, meExternalSbHints, meSbHintsCount, stateBufferIdx,
        outputReconBuffer, reserved3, reserved6,
    });
    layout!(checks, enc::NV_ENC_LOCK_BITSTREAM, "NV_ENC_LOCK_BITSTREAM" {
        version, outputBitstream, sliceOffsets, frameIdx, hwEncodeStatus, numSlices,
        bitstreamSizeInBytes, outputTimeStamp, outputDuration, bitstreamBufferPtr, pictureType,
        pictureStruct, frameAvgQP, frameSatd, ltrFrameIdx, ltrFrameBitmap, temporalId,
        intraMBCount, interMBCount, averageMVX, averageMVY, alphaLayerSizeInBytes,
        outputStatsPtrSize, reserved, outputStatsPtr, frameIdxDisplay, reserved1, reserved2,
        reservedInternal,
    });
    bit_fields!(checks, enc::NV_ENC_LOCK_BITSTREAM, "NV_ENC_LOCK_BITSTREAM", flags: u32,
    enc::bits::lock { doNotWait });
    layout!(checks, enc::NV_ENC_LOCK_INPUT_BUFFER, "NV_ENC_LOCK_INPUT_BUFFER" {
        version, inputBuffer, bufferDataPtr, pitch, reserved1, reserved2,
    });
    layout!(checks, enc::NV_ENC_SEQUENCE_PARAM_PAYLOAD, "NV_ENC_SEQUENCE_PARAM_PAYLOAD" {
        version, inBufferSize, spsId, ppsId, spsppsBuffer, outSPSPPSPayloadSize, reserved,
        reserved2,
    });
    layout!(checks, enc::NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS,
        "NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS" {
        version, deviceType, device, reserved, apiVersion, reserved1, reserved2,
    });
    layout!(checks, enc::NV_ENCODE_API_FUNCTION_LIST, "NV_ENCODE_API_FUNCTION_LIST" {
        version, reserved, nvEncOpenEncodeSession, nvEncGetEncodeGUIDCount,
        nvEncGetEncodeProfileGUIDCount, nvEncGetEncodeProfileGUIDs, nvEncGetEncodeGUIDs,
        nvEncGetInputFormatCount, nvEncGetInputFormats, nvEncGetEncodeCaps,
        nvEncGetEncodePresetCount, nvEncGetEncodePresetGUIDs, nvEncGetEncodePresetConfig,
        nvEncInitializeEncoder, nvEncCreateInputBuffer, nvEncDestroyInputBuffer,
        nvEncCreateBitstreamBuffer, nvEncDestroyBitstreamBuffer, nvEncEncodePicture,
        nvEncLockBitstream, nvEncUnlockBitstream, nvEncLockInputBuffer, nvEncUnlockInputBuffer,
        nvEncGetEncodeStats, nvEncGetSequenceParams, nvEncRegisterAsyncEvent,
        nvEncUnregisterAsyncEvent, nvEncMapInputResource, nvEncUnmapInputResource,
        nvEncDestroyEncoder, nvEncInvalidateRefFrames, nvEncOpenEncodeSessionEx,
        nvEncRegisterResource, nvEncUnregisterResource, nvEncReconfigureEncoder, reserved1,
        nvEncCreateMVBuffer, nvEncDestroyMVBuffer, nvEncRunMotionEstimationOnly,
        nvEncGetLastErrorString, nvEncSetIOCudaStreams, nvEncGetEncodePresetConfigEx,
        nvEncGetSequenceParamEx, nvEncRestoreEncoderState, nvEncLookaheadPicture, reserved2,
    });

    for (name, value) in [
        ("NVENCAPI_MAJOR_VERSION", enc::NVENCAPI_MAJOR_VERSION),
        ("NVENCAPI_MINOR_VERSION", enc::NVENCAPI_MINOR_VERSION),
        ("NVENCAPI_VERSION", enc::NVENCAPI_VERSION),
        ("NV_ENC_CAPS_PARAM_VER", enc::NV_ENC_CAPS_PARAM_VER),
        (
            "NV_ENC_CREATE_INPUT_BUFFER_VER",
            enc::NV_ENC_CREATE_INPUT_BUFFER_VER,
        ),
        (
            "NV_ENC_CREATE_BITSTREAM_BUFFER_VER",
            enc::NV_ENC_CREATE_BITSTREAM_BUFFER_VER,
        ),
        ("NV_ENC_RC_PARAMS_VER", enc::NV_ENC_RC_PARAMS_VER),
        ("NV_ENC_CONFIG_VER", enc::NV_ENC_CONFIG_VER),
        (
            "NV_ENC_INITIALIZE_PARAMS_VER",
            enc::NV_ENC_INITIALIZE_PARAMS_VER,
        ),
        (
            "NV_ENC_RECONFIGURE_PARAMS_VER",
            enc::NV_ENC_RECONFIGURE_PARAMS_VER,
        ),
        ("NV_ENC_PRESET_CONFIG_VER", enc::NV_ENC_PRESET_CONFIG_VER),
        ("NV_ENC_PIC_PARAMS_VER", enc::NV_ENC_PIC_PARAMS_VER),
        ("NV_ENC_LOCK_BITSTREAM_VER", enc::NV_ENC_LOCK_BITSTREAM_VER),
        (
            "NV_ENC_LOCK_INPUT_BUFFER_VER",
            enc::NV_ENC_LOCK_INPUT_BUFFER_VER,
        ),
        (
            "NV_ENC_SEQUENCE_PARAM_PAYLOAD_VER",
            enc::NV_ENC_SEQUENCE_PARAM_PAYLOAD_VER,
        ),
        (
            "NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER",
            enc::NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER,
        ),
        (
            "NV_ENCODE_API_FUNCTION_LIST_VER",
            enc::NV_ENCODE_API_FUNCTION_LIST_VER,
        ),
        ("NVENC_INFINITE_GOPLENGTH", enc::NVENC_INFINITE_GOPLENGTH),
        ("NV_ENC_PIC_FLAG_FORCEIDR", enc::NV_ENC_PIC_FLAG_FORCEIDR),
        (
            "NV_ENC_PIC_FLAG_OUTPUT_SPSPPS",
            enc::NV_ENC_PIC_FLAG_OUTPUT_SPSPPS,
        ),
        ("NV_ENC_PIC_FLAG_EOS", enc::NV_ENC_PIC_FLAG_EOS),
        ("NV_ENC_LEVEL_AUTOSELECT", enc::NV_ENC_LEVEL_AUTOSELECT),
    ] {
        checks.constant(name, value.into());
    }
    for (name, value) in [
        ("NV_ENC_SUCCESS", enc::NV_ENC_SUCCESS),
        (
            "NV_ENC_ERR_NEED_MORE_INPUT",
            enc::NV_ENC_ERR_NEED_MORE_INPUT,
        ),
        ("NV_ENC_DEVICE_TYPE_CUDA", enc::NV_ENC_DEVICE_TYPE_CUDA),
        ("NV_ENC_BUFFER_FORMAT_NV12", enc::NV_ENC_BUFFER_FORMAT_NV12),
        (
            "NV_ENC_BUFFER_FORMAT_YUV420_10BIT",
            enc::NV_ENC_BUFFER_FORMAT_YUV420_10BIT,
        ),
        (
            "NV_ENC_MEMORY_HEAP_AUTOSELECT",
            enc::NV_ENC_MEMORY_HEAP_AUTOSELECT,
        ),
        (
            "NV_ENC_TUNING_INFO_HIGH_QUALITY",
            enc::NV_ENC_TUNING_INFO_HIGH_QUALITY,
        ),
        (
            "NV_ENC_TUNING_INFO_LOW_LATENCY",
            enc::NV_ENC_TUNING_INFO_LOW_LATENCY,
        ),
        ("NV_ENC_PARAMS_RC_CONSTQP", enc::NV_ENC_PARAMS_RC_CONSTQP),
        ("NV_ENC_PARAMS_RC_VBR", enc::NV_ENC_PARAMS_RC_VBR),
        ("NV_ENC_PARAMS_RC_CBR", enc::NV_ENC_PARAMS_RC_CBR),
        ("NV_ENC_PIC_STRUCT_FRAME", enc::NV_ENC_PIC_STRUCT_FRAME),
        ("NV_ENC_PIC_TYPE_P", enc::NV_ENC_PIC_TYPE_P),
        ("NV_ENC_PIC_TYPE_B", enc::NV_ENC_PIC_TYPE_B),
        ("NV_ENC_PIC_TYPE_I", enc::NV_ENC_PIC_TYPE_I),
        ("NV_ENC_PIC_TYPE_IDR", enc::NV_ENC_PIC_TYPE_IDR),
        ("NV_ENC_PIC_TYPE_NONREF_P", enc::NV_ENC_PIC_TYPE_NONREF_P),
        (
            "NV_ENC_H264_ENTROPY_CODING_MODE_CABAC",
            enc::NV_ENC_H264_ENTROPY_CODING_MODE_CABAC,
        ),
        (
            "NV_ENC_H264_ENTROPY_CODING_MODE_CAVLC",
            enc::NV_ENC_H264_ENTROPY_CODING_MODE_CAVLC,
        ),
        (
            "NV_ENC_BFRAME_REF_MODE_DISABLED",
            enc::NV_ENC_BFRAME_REF_MODE_DISABLED,
        ),
        ("NV_ENC_BIT_DEPTH_8", enc::NV_ENC_BIT_DEPTH_8),
        ("NV_ENC_BIT_DEPTH_10", enc::NV_ENC_BIT_DEPTH_10),
        (
            "NV_ENC_VUI_VIDEO_FORMAT_UNSPECIFIED",
            enc::NV_ENC_VUI_VIDEO_FORMAT_UNSPECIFIED,
        ),
        ("NV_ENC_CAPS_NUM_MAX_BFRAMES", enc::caps::NUM_MAX_BFRAMES),
        (
            "NV_ENC_CAPS_SUPPORTED_RATECONTROL_MODES",
            enc::caps::SUPPORTED_RATECONTROL_MODES,
        ),
        ("NV_ENC_CAPS_SUPPORT_CABAC", enc::caps::SUPPORT_CABAC),
        ("NV_ENC_CAPS_WIDTH_MAX", enc::caps::WIDTH_MAX),
        ("NV_ENC_CAPS_HEIGHT_MAX", enc::caps::HEIGHT_MAX),
        (
            "NV_ENC_CAPS_SUPPORT_10BIT_ENCODE",
            enc::caps::SUPPORT_10BIT_ENCODE,
        ),
        ("NV_ENC_CAPS_WIDTH_MIN", enc::caps::WIDTH_MIN),
        ("NV_ENC_CAPS_HEIGHT_MIN", enc::caps::HEIGHT_MIN),
        (
            "NV_ENC_CAPS_SUPPORT_MULTIPLE_REF_FRAMES",
            enc::caps::SUPPORT_MULTIPLE_REF_FRAMES,
        ),
    ] {
        checks.constant(name, value.into());
    }
    for (value, name) in enc::CAPS_NAMES.iter().enumerate() {
        checks.constant(
            &format!("NV_ENC_CAPS_{name}"),
            i64::try_from(value).unwrap_or(-1),
        );
    }
    for (value, name) in enc::BUFFER_FORMAT_NAMES {
        checks.constant(&format!("NV_ENC_BUFFER_FORMAT_{name}"), value.into());
    }
    for (value, name) in enc::NVENCSTATUS_NAMES.iter().enumerate() {
        checks.constant(name, i64::try_from(value).unwrap_or(-1));
    }
    for (name, guid) in [
        ("NV_ENC_CODEC_H264_GUID", enc::NV_ENC_CODEC_H264_GUID),
        ("NV_ENC_CODEC_HEVC_GUID", enc::NV_ENC_CODEC_HEVC_GUID),
        ("NV_ENC_CODEC_AV1_GUID", enc::NV_ENC_CODEC_AV1_GUID),
        (
            "NV_ENC_CODEC_PROFILE_AUTOSELECT_GUID",
            enc::NV_ENC_CODEC_PROFILE_AUTOSELECT_GUID,
        ),
        (
            "NV_ENC_H264_PROFILE_BASELINE_GUID",
            enc::NV_ENC_H264_PROFILE_BASELINE_GUID,
        ),
        (
            "NV_ENC_H264_PROFILE_MAIN_GUID",
            enc::NV_ENC_H264_PROFILE_MAIN_GUID,
        ),
        (
            "NV_ENC_H264_PROFILE_HIGH_GUID",
            enc::NV_ENC_H264_PROFILE_HIGH_GUID,
        ),
        (
            "NV_ENC_H264_PROFILE_HIGH_10_GUID",
            enc::NV_ENC_H264_PROFILE_HIGH_10_GUID,
        ),
        (
            "NV_ENC_H264_PROFILE_HIGH_444_GUID",
            enc::NV_ENC_H264_PROFILE_HIGH_444_GUID,
        ),
        (
            "NV_ENC_H264_PROFILE_CONSTRAINED_HIGH_GUID",
            enc::NV_ENC_H264_PROFILE_CONSTRAINED_HIGH_GUID,
        ),
        (
            "NV_ENC_HEVC_PROFILE_MAIN_GUID",
            enc::NV_ENC_HEVC_PROFILE_MAIN_GUID,
        ),
        (
            "NV_ENC_HEVC_PROFILE_MAIN10_GUID",
            enc::NV_ENC_HEVC_PROFILE_MAIN10_GUID,
        ),
        (
            "NV_ENC_HEVC_PROFILE_FREXT_GUID",
            enc::NV_ENC_HEVC_PROFILE_FREXT_GUID,
        ),
        (
            "NV_ENC_AV1_PROFILE_MAIN_GUID",
            enc::NV_ENC_AV1_PROFILE_MAIN_GUID,
        ),
        ("NV_ENC_PRESET_P1_GUID", enc::NV_ENC_PRESET_P1_GUID),
        ("NV_ENC_PRESET_P2_GUID", enc::NV_ENC_PRESET_P2_GUID),
        ("NV_ENC_PRESET_P3_GUID", enc::NV_ENC_PRESET_P3_GUID),
        ("NV_ENC_PRESET_P4_GUID", enc::NV_ENC_PRESET_P4_GUID),
        ("NV_ENC_PRESET_P5_GUID", enc::NV_ENC_PRESET_P5_GUID),
        ("NV_ENC_PRESET_P6_GUID", enc::NV_ENC_PRESET_P6_GUID),
        ("NV_ENC_PRESET_P7_GUID", enc::NV_ENC_PRESET_P7_GUID),
    ] {
        checks.guid(name, guid);
    }
    // The entries taking structs this binding does not declare are passed
    // as `void *` here and never called; their places in the list are
    // checked above. reserved1 is not an entry point.
    const NOT_CALLED: [&str; 10] = [
        "reserved1",
        "nvEncGetEncodeStats",
        "nvEncRegisterAsyncEvent",
        "nvEncUnregisterAsyncEvent",
        "nvEncMapInputResource",
        "nvEncRegisterResource",
        "nvEncCreateMVBuffer",
        "nvEncRunMotionEstimationOnly",
        "nvEncRestoreEncoderState",
        "nvEncLookaheadPicture",
    ];
    for signature in enc::NV_ENCODE_API_FUNCTION_LIST::SIGNATURES {
        if !NOT_CALLED.contains(&signature.symbol) {
            let field = format!("((NV_ENCODE_API_FUNCTION_LIST *)0)->{}", signature.symbol);
            checks.signature(&field, signature, true);
        }
    }
    for signature in enc::NvencEntryPoints::SIGNATURES {
        checks.signature(signature.symbol, signature, false);
    }
}

fn all_checks() -> Checks {
    let mut checks = Checks::default();
    cuda_checks(&mut checks);
    nvdec_checks(&mut checks);
    nvenc_checks(&mut checks);
    checks
}

/// The C program: the headers, a byte dumper for bit-fields, and every
/// check's statement.
fn program(checks: &Checks) -> String {
    let mut c = String::from(
        "#include <stddef.h>\n#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n\
         #include <ffnvcodec/dynlink_cuda.h>\n#include <ffnvcodec/dynlink_cuviddec.h>\n\
         #include <ffnvcodec/nvEncodeAPI.h>\n\
         static void dump(const char *name, const unsigned char *p, size_t n) {\n\
         \tprintf(\"%s\", name);\n\
         \tfor (size_t i = 0; i < n; i++) if (p[i]) printf(\" %zu=%02x\", i, p[i]);\n\
         \tprintf(\"\\n\");\n}\nint main(void) {\n",
    );
    for check in &checks.0 {
        c.push('\t');
        c.push_str(&check.c);
        c.push('\n');
    }
    c.push_str("\treturn 0;\n}\n");
    c
}

#[test]
fn the_c_spelling_of_rust_types() {
    assert_eq!(c_of("*mut *const c_char"), "char const * *");
    assert_eq!(c_of("* mut Encoder"), "void * *");
    assert_eq!(c_of("*const CUDA_MEMCPY2D"), "CUDA_MEMCPY2D const *");
    assert_eq!(c_of("c_uint"), "unsigned int");
}

#[test]
fn every_check_has_a_c_statement() {
    let checks = all_checks();
    assert!(checks.0.len() > 500);
    assert!(program(&checks).contains("sizeof(CUVIDPICPARAMS)"));
}

/// Compiles the checks against the installed headers (`CC`, `cc` by
/// default, with the headers in its include path) and compares what the C
/// program prints with what Rust declares.
#[test]
#[ignore = "needs ffnvcodec's headers and a C compiler: run on the NVIDIA host"]
fn layouts_match_nvidia_headers() -> Result<(), Box<dyn std::error::Error>> {
    let checks = all_checks();
    let dir = std::env::temp_dir().join(format!("rvgv-nvidia-layout-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let binary = dir.join("layout");
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    let mut compile = Command::new(cc)
        .args(["-Wno-overflow", "-x", "c", "-", "-o"])
        .arg(&binary)
        .stdin(Stdio::piped())
        .spawn()?;
    compile
        .stdin
        .take()
        .ok_or("no stdin")?
        .write_all(program(&checks).as_bytes())?;
    let compiled = compile.wait()?.success();
    let output = if compiled {
        Some(Command::new(&binary).output()?)
    } else {
        None
    };
    std::fs::remove_dir_all(&dir)?;
    let output = output.ok_or("the C compiler failed")?;
    let c_lines: Vec<String> = String::from_utf8(output.stdout)?
        .lines()
        .map(str::to_owned)
        .collect();
    let rust_lines: Vec<String> = checks.0.into_iter().map(|check| check.expected).collect();
    let differences: Vec<String> = rust_lines
        .iter()
        .zip(&c_lines)
        .filter(|(rust, c)| rust != c)
        .map(|(rust, c)| format!("rust: {rust}\n   c: {c}"))
        .collect();
    assert!(
        differences.is_empty() && rust_lines.len() == c_lines.len(),
        "{} of {} lines differ ({} printed):\n{}",
        differences.len(),
        rust_lines.len(),
        c_lines.len(),
        differences.join("\n")
    );
    Ok(())
}
