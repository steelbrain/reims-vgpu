//! NVENC's declarations (`nvEncodeAPI.h`), the subset this crate uses, at
//! NVENC API 13.1: the function list, sessions on a CUDA context, the codec,
//! profile and preset GUIDs, caps, the configuration and initialisation,
//! input and bitstream buffers, pictures, and the sequence parameters.
//!
//! Field names are the header's, so that `crate::nvidia::layout`'s test can check
//! each size and offset against ffnvcodec's `nvEncodeAPI.h` (13.1.15.0).
//! Each versioned struct's version constant is checked there too, and the
//! driver refuses a wrong one with `NV_ENC_ERR_INVALID_VERSION`. A C
//! bit-field word is one Rust `u32`, named `flags`, with its fields'
//! positions in [`bits`]; those names are ours, each marked so.
//!
//! The members this crate never sets are kept as their bytes: the AV1
//! configuration, the codecs' picture parameters, motion estimation hints.

#![allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    clippy::upper_case_acronyms,
    reason = "NVIDIA's names, kept for grepping the headers and for the layout check"
)]

use std::ffi::{c_char, c_int, c_void};

pub(crate) use crate::nvidia::dl::Zeroed;

/// `NVENCSTATUS`: 0 is success.
pub(crate) type NVENCSTATUS = c_int;
/// `NV_ENC_CAPS`.
pub(crate) type NV_ENC_CAPS = c_int;
/// `NV_ENC_BUFFER_FORMAT`.
pub(crate) type NV_ENC_BUFFER_FORMAT = c_int;
/// `NV_ENC_TUNING_INFO`.
pub(crate) type NV_ENC_TUNING_INFO = c_int;
/// `NV_ENC_PARAMS_RC_MODE`.
pub(crate) type NV_ENC_PARAMS_RC_MODE = c_int;
/// `NV_ENC_PIC_TYPE`.
pub(crate) type NV_ENC_PIC_TYPE = c_int;
/// `NV_ENC_PIC_STRUCT`.
pub(crate) type NV_ENC_PIC_STRUCT = c_int;
/// The `NV_ENC_*` enums this crate passes through without naming their
/// values: one alias, since each is a C enum the size of an `int`. Name is
/// ours.
pub(crate) type NvEncEnum = c_int;

/// `NVENCAPI_MAJOR_VERSION`.
pub(crate) const NVENCAPI_MAJOR_VERSION: u32 = 13;
/// `NVENCAPI_MINOR_VERSION`.
pub(crate) const NVENCAPI_MINOR_VERSION: u32 = 1;
/// `NVENCAPI_VERSION`: the API version a session opens with.
pub(crate) const NVENCAPI_VERSION: u32 = NVENCAPI_MAJOR_VERSION | (NVENCAPI_MINOR_VERSION << 24);

/// `NVENCAPI_STRUCT_VERSION(ver)`.
const fn struct_version(version: u32) -> u32 {
    NVENCAPI_VERSION | (version << 16) | (0x7 << 28)
}

pub(crate) const NV_ENC_CAPS_PARAM_VER: u32 = struct_version(1);
pub(crate) const NV_ENC_CREATE_INPUT_BUFFER_VER: u32 = struct_version(2);
pub(crate) const NV_ENC_CREATE_BITSTREAM_BUFFER_VER: u32 = struct_version(1);
pub(crate) const NV_ENC_RC_PARAMS_VER: u32 = struct_version(1);
pub(crate) const NV_ENC_CONFIG_VER: u32 = struct_version(9) | (1 << 31);
pub(crate) const NV_ENC_INITIALIZE_PARAMS_VER: u32 = struct_version(7) | (1 << 31);
pub(crate) const NV_ENC_RECONFIGURE_PARAMS_VER: u32 = struct_version(2) | (1 << 31);
pub(crate) const NV_ENC_PRESET_CONFIG_VER: u32 = struct_version(5) | (1 << 31);
pub(crate) const NV_ENC_PIC_PARAMS_VER: u32 = struct_version(7) | (1 << 31);
pub(crate) const NV_ENC_LOCK_BITSTREAM_VER: u32 = struct_version(2) | (1 << 31);
pub(crate) const NV_ENC_LOCK_INPUT_BUFFER_VER: u32 = struct_version(1);
pub(crate) const NV_ENC_SEQUENCE_PARAM_PAYLOAD_VER: u32 = struct_version(1);
pub(crate) const NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER: u32 = struct_version(1);
pub(crate) const NV_ENCODE_API_FUNCTION_LIST_VER: u32 = struct_version(2);

/// `NVENC_INFINITE_GOPLENGTH`.
pub(crate) const NVENC_INFINITE_GOPLENGTH: u32 = 0xffff_ffff;

pub(crate) const NV_ENC_SUCCESS: NVENCSTATUS = 0;
pub(crate) const NV_ENC_ERR_NEED_MORE_INPUT: NVENCSTATUS = 17;

/// `NVENCSTATUS`'s names, by value.
pub(crate) const NVENCSTATUS_NAMES: [&str; 27] = [
    "NV_ENC_SUCCESS",
    "NV_ENC_ERR_NO_ENCODE_DEVICE",
    "NV_ENC_ERR_UNSUPPORTED_DEVICE",
    "NV_ENC_ERR_INVALID_ENCODERDEVICE",
    "NV_ENC_ERR_INVALID_DEVICE",
    "NV_ENC_ERR_DEVICE_NOT_EXIST",
    "NV_ENC_ERR_INVALID_PTR",
    "NV_ENC_ERR_INVALID_EVENT",
    "NV_ENC_ERR_INVALID_PARAM",
    "NV_ENC_ERR_INVALID_CALL",
    "NV_ENC_ERR_OUT_OF_MEMORY",
    "NV_ENC_ERR_ENCODER_NOT_INITIALIZED",
    "NV_ENC_ERR_UNSUPPORTED_PARAM",
    "NV_ENC_ERR_LOCK_BUSY",
    "NV_ENC_ERR_NOT_ENOUGH_BUFFER",
    "NV_ENC_ERR_INVALID_VERSION",
    "NV_ENC_ERR_MAP_FAILED",
    "NV_ENC_ERR_NEED_MORE_INPUT",
    "NV_ENC_ERR_ENCODER_BUSY",
    "NV_ENC_ERR_EVENT_NOT_REGISTERD",
    "NV_ENC_ERR_GENERIC",
    "NV_ENC_ERR_INCOMPATIBLE_CLIENT_KEY",
    "NV_ENC_ERR_UNIMPLEMENTED",
    "NV_ENC_ERR_RESOURCE_REGISTER_FAILED",
    "NV_ENC_ERR_RESOURCE_NOT_REGISTERED",
    "NV_ENC_ERR_RESOURCE_NOT_MAPPED",
    "NV_ENC_ERR_NEED_MORE_OUTPUT",
];

/// `NV_ENC_DEVICE_TYPE_CUDA`.
pub(crate) const NV_ENC_DEVICE_TYPE_CUDA: NvEncEnum = 1;

pub(crate) const NV_ENC_BUFFER_FORMAT_NV12: NV_ENC_BUFFER_FORMAT = 0x1;
pub(crate) const NV_ENC_BUFFER_FORMAT_YUV420_10BIT: NV_ENC_BUFFER_FORMAT = 0x1_0000;

pub(crate) const NV_ENC_MEMORY_HEAP_AUTOSELECT: NvEncEnum = 0;

pub(crate) const NV_ENC_TUNING_INFO_HIGH_QUALITY: NV_ENC_TUNING_INFO = 1;
pub(crate) const NV_ENC_TUNING_INFO_LOW_LATENCY: NV_ENC_TUNING_INFO = 2;

pub(crate) const NV_ENC_PARAMS_RC_CONSTQP: NV_ENC_PARAMS_RC_MODE = 0x0;
pub(crate) const NV_ENC_PARAMS_RC_VBR: NV_ENC_PARAMS_RC_MODE = 0x1;
pub(crate) const NV_ENC_PARAMS_RC_CBR: NV_ENC_PARAMS_RC_MODE = 0x2;

pub(crate) const NV_ENC_PIC_STRUCT_FRAME: NV_ENC_PIC_STRUCT = 0x1;

pub(crate) const NV_ENC_PIC_TYPE_P: NV_ENC_PIC_TYPE = 0x0;
pub(crate) const NV_ENC_PIC_TYPE_B: NV_ENC_PIC_TYPE = 0x1;
pub(crate) const NV_ENC_PIC_TYPE_I: NV_ENC_PIC_TYPE = 0x2;
pub(crate) const NV_ENC_PIC_TYPE_IDR: NV_ENC_PIC_TYPE = 0x3;
pub(crate) const NV_ENC_PIC_TYPE_NONREF_P: NV_ENC_PIC_TYPE = 0x7;

pub(crate) const NV_ENC_PIC_FLAG_FORCEIDR: u32 = 0x2;
pub(crate) const NV_ENC_PIC_FLAG_OUTPUT_SPSPPS: u32 = 0x4;
pub(crate) const NV_ENC_PIC_FLAG_EOS: u32 = 0x8;

pub(crate) const NV_ENC_H264_ENTROPY_CODING_MODE_CABAC: NvEncEnum = 0x1;
pub(crate) const NV_ENC_H264_ENTROPY_CODING_MODE_CAVLC: NvEncEnum = 0x2;

pub(crate) const NV_ENC_BFRAME_REF_MODE_DISABLED: NvEncEnum = 0x0;

pub(crate) const NV_ENC_BIT_DEPTH_8: NvEncEnum = 8;
pub(crate) const NV_ENC_BIT_DEPTH_10: NvEncEnum = 10;

pub(crate) const NV_ENC_LEVEL_AUTOSELECT: u32 = 0;

/// The `NV_ENC_CAPS` values this crate asks for.
pub(crate) mod caps {
    use super::NV_ENC_CAPS;
    pub(crate) const NUM_MAX_BFRAMES: NV_ENC_CAPS = 0;
    pub(crate) const SUPPORTED_RATECONTROL_MODES: NV_ENC_CAPS = 1;
    pub(crate) const SUPPORT_CABAC: NV_ENC_CAPS = 7;
    pub(crate) const WIDTH_MAX: NV_ENC_CAPS = 16;
    pub(crate) const HEIGHT_MAX: NV_ENC_CAPS = 17;
    pub(crate) const SUPPORT_10BIT_ENCODE: NV_ENC_CAPS = 39;
    pub(crate) const WIDTH_MIN: NV_ENC_CAPS = 45;
    pub(crate) const HEIGHT_MIN: NV_ENC_CAPS = 46;
    pub(crate) const SUPPORT_MULTIPLE_REF_FRAMES: NV_ENC_CAPS = 47;
}

/// `NV_ENC_CAPS`' names without their `NV_ENC_CAPS_` prefix, by value,
/// up to `NV_ENC_CAPS_EXPOSED_COUNT`.
pub(crate) const CAPS_NAMES: [&str; 60] = [
    "NUM_MAX_BFRAMES",
    "SUPPORTED_RATECONTROL_MODES",
    "SUPPORT_FIELD_ENCODING",
    "SUPPORT_MONOCHROME",
    "SUPPORT_FMO",
    "SUPPORT_QPELMV",
    "SUPPORT_BDIRECT_MODE",
    "SUPPORT_CABAC",
    "SUPPORT_ADAPTIVE_TRANSFORM",
    "SUPPORT_STEREO_MVC",
    "NUM_MAX_TEMPORAL_LAYERS",
    "SUPPORT_HIERARCHICAL_PFRAMES",
    "SUPPORT_HIERARCHICAL_BFRAMES",
    "LEVEL_MAX",
    "LEVEL_MIN",
    "SEPARATE_COLOUR_PLANE",
    "WIDTH_MAX",
    "HEIGHT_MAX",
    "SUPPORT_TEMPORAL_SVC",
    "SUPPORT_DYN_RES_CHANGE",
    "SUPPORT_DYN_BITRATE_CHANGE",
    "SUPPORT_DYN_FORCE_CONSTQP",
    "SUPPORT_DYN_RCMODE_CHANGE",
    "SUPPORT_SUBFRAME_READBACK",
    "SUPPORT_CONSTRAINED_ENCODING",
    "SUPPORT_INTRA_REFRESH",
    "SUPPORT_CUSTOM_VBV_BUF_SIZE",
    "SUPPORT_DYNAMIC_SLICE_MODE",
    "SUPPORT_REF_PIC_INVALIDATION",
    "PREPROC_SUPPORT",
    "ASYNC_ENCODE_SUPPORT",
    "MB_NUM_MAX",
    "MB_PER_SEC_MAX",
    "SUPPORT_YUV444_ENCODE",
    "SUPPORT_LOSSLESS_ENCODE",
    "SUPPORT_SAO",
    "SUPPORT_MEONLY_MODE",
    "SUPPORT_LOOKAHEAD",
    "SUPPORT_TEMPORAL_AQ",
    "SUPPORT_10BIT_ENCODE",
    "NUM_MAX_LTR_FRAMES",
    "SUPPORT_WEIGHTED_PREDICTION",
    "DYNAMIC_QUERY_ENCODER_CAPACITY",
    "SUPPORT_BFRAME_REF_MODE",
    "SUPPORT_EMPHASIS_LEVEL_MAP",
    "WIDTH_MIN",
    "HEIGHT_MIN",
    "SUPPORT_MULTIPLE_REF_FRAMES",
    "SUPPORT_ALPHA_LAYER_ENCODING",
    "NUM_ENCODER_ENGINES",
    "SINGLE_SLICE_INTRA_REFRESH",
    "DISABLE_ENC_STATE_ADVANCE",
    "OUTPUT_RECON_SURFACE",
    "OUTPUT_BLOCK_STATS",
    "OUTPUT_ROW_STATS",
    "SUPPORT_TEMPORAL_FILTER",
    "SUPPORT_LOOKAHEAD_LEVEL",
    "SUPPORT_UNIDIRECTIONAL_B",
    "SUPPORT_MVHEVC_ENCODE",
    "SUPPORT_YUV422_ENCODE",
];

/// `NV_ENC_BUFFER_FORMAT`'s names without their prefix.
pub(crate) const BUFFER_FORMAT_NAMES: [(NV_ENC_BUFFER_FORMAT, &str); 14] = [
    (0x0000_0001, "NV12"),
    (0x0000_0010, "YV12"),
    (0x0000_0100, "IYUV"),
    (0x0000_1000, "YUV444"),
    (0x0001_0000, "YUV420_10BIT"),
    (0x0010_0000, "YUV444_10BIT"),
    (0x0100_0000, "ARGB"),
    (0x0200_0000, "ARGB10"),
    (0x0400_0000, "AYUV"),
    (0x1000_0000, "ABGR"),
    (0x2000_0000, "ABGR10"),
    (0x4000_0000, "U8"),
    (0x4000_0001, "NV16"),
    (0x4000_0002, "P210"),
];

/// `GUID`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct GUID {
    pub(crate) Data1: u32,
    pub(crate) Data2: u16,
    pub(crate) Data3: u16,
    pub(crate) Data4: [u8; 8],
}

const fn guid(data1: u32, data2: u16, data3: u16, data4: [u8; 8]) -> GUID {
    GUID {
        Data1: data1,
        Data2: data2,
        Data3: data3,
        Data4: data4,
    }
}

pub(crate) const NV_ENC_CODEC_H264_GUID: GUID = guid(
    0x6bc8_2762,
    0x4e63,
    0x4ca4,
    [0xaa, 0x85, 0x1e, 0x50, 0xf3, 0x21, 0xf6, 0xbf],
);
pub(crate) const NV_ENC_CODEC_HEVC_GUID: GUID = guid(
    0x790c_dc88,
    0x4522,
    0x4d7b,
    [0x94, 0x25, 0xbd, 0xa9, 0x97, 0x5f, 0x76, 0x03],
);
pub(crate) const NV_ENC_CODEC_AV1_GUID: GUID = guid(
    0x0a35_2289,
    0x0aa7,
    0x4759,
    [0x86, 0x2d, 0x5d, 0x15, 0xcd, 0x16, 0xd2, 0x54],
);
pub(crate) const NV_ENC_CODEC_PROFILE_AUTOSELECT_GUID: GUID = guid(
    0xbfd6_f8e7,
    0x233c,
    0x4341,
    [0x8b, 0x3e, 0x48, 0x18, 0x52, 0x38, 0x03, 0xf4],
);
pub(crate) const NV_ENC_H264_PROFILE_BASELINE_GUID: GUID = guid(
    0x0727_bcaa,
    0x78c4,
    0x4c83,
    [0x8c, 0x2f, 0xef, 0x3d, 0xff, 0x26, 0x7c, 0x6a],
);
pub(crate) const NV_ENC_H264_PROFILE_MAIN_GUID: GUID = guid(
    0x60b5_c1d4,
    0x67fe,
    0x4790,
    [0x94, 0xd5, 0xc4, 0x72, 0x6d, 0x7b, 0x6e, 0x6d],
);
pub(crate) const NV_ENC_H264_PROFILE_HIGH_GUID: GUID = guid(
    0xe7cb_c309,
    0x4f7a,
    0x4b89,
    [0xaf, 0x2a, 0xd5, 0x37, 0xc9, 0x2b, 0xe3, 0x10],
);
pub(crate) const NV_ENC_H264_PROFILE_HIGH_10_GUID: GUID = guid(
    0x8f0c_337e,
    0x186c,
    0x48e9,
    [0xa6, 0x9d, 0x7a, 0x83, 0x34, 0x08, 0x97, 0x58],
);
pub(crate) const NV_ENC_H264_PROFILE_HIGH_444_GUID: GUID = guid(
    0x7ac6_63cb,
    0xa598,
    0x4960,
    [0xb8, 0x44, 0x33, 0x9b, 0x26, 0x1a, 0x7d, 0x52],
);
pub(crate) const NV_ENC_H264_PROFILE_CONSTRAINED_HIGH_GUID: GUID = guid(
    0xaec1_bd87,
    0xe85b,
    0x48f2,
    [0x84, 0xc3, 0x98, 0xbc, 0xa6, 0x28, 0x50, 0x72],
);
pub(crate) const NV_ENC_HEVC_PROFILE_MAIN_GUID: GUID = guid(
    0xb514_c39a,
    0xb55b,
    0x40fa,
    [0x87, 0x8f, 0xf1, 0x25, 0x3b, 0x4d, 0xfd, 0xec],
);
pub(crate) const NV_ENC_HEVC_PROFILE_MAIN10_GUID: GUID = guid(
    0xfa4d_2b6c,
    0x3a5b,
    0x411a,
    [0x80, 0x18, 0x0a, 0x3f, 0x5e, 0x3c, 0x9b, 0xe5],
);
pub(crate) const NV_ENC_HEVC_PROFILE_FREXT_GUID: GUID = guid(
    0x51ec_32b5,
    0x1b4c,
    0x453c,
    [0x9c, 0xbd, 0xb6, 0x16, 0xbd, 0x62, 0x13, 0x41],
);
pub(crate) const NV_ENC_AV1_PROFILE_MAIN_GUID: GUID = guid(
    0x5f2a_39f5,
    0xf14e,
    0x4f95,
    [0x9a, 0x9e, 0xb7, 0x6d, 0x56, 0x8f, 0xcf, 0x97],
);
pub(crate) const NV_ENC_PRESET_P1_GUID: GUID = guid(
    0xfc0a_8d3e,
    0x45f8,
    0x4cf8,
    [0x80, 0xc7, 0x29, 0x88, 0x71, 0x59, 0x0e, 0xbf],
);
pub(crate) const NV_ENC_PRESET_P2_GUID: GUID = guid(
    0xf581_cfb8,
    0x88d6,
    0x4381,
    [0x93, 0xf0, 0xdf, 0x13, 0xf9, 0xc2, 0x7d, 0xab],
);
pub(crate) const NV_ENC_PRESET_P3_GUID: GUID = guid(
    0x3685_0110,
    0x3a07,
    0x441f,
    [0x94, 0xd5, 0x36, 0x70, 0x63, 0x1f, 0x91, 0xf6],
);
pub(crate) const NV_ENC_PRESET_P4_GUID: GUID = guid(
    0x90a7_b826,
    0xdf06,
    0x4862,
    [0xb9, 0xd2, 0xcd, 0x6d, 0x73, 0xa0, 0x86, 0x81],
);
pub(crate) const NV_ENC_PRESET_P5_GUID: GUID = guid(
    0x21c6_e6b4,
    0x297a,
    0x4cba,
    [0x99, 0x8f, 0xb6, 0xcb, 0xde, 0x72, 0xad, 0xe3],
);
pub(crate) const NV_ENC_PRESET_P6_GUID: GUID = guid(
    0x8e75_c279,
    0x6299,
    0x4ab6,
    [0x83, 0x02, 0x0b, 0x21, 0x5a, 0x33, 0x5c, 0xf5],
);
pub(crate) const NV_ENC_PRESET_P7_GUID: GUID = guid(
    0x8484_8c12,
    0x6f71,
    0x4c13,
    [0x93, 0x1b, 0x53, 0xe2, 0x83, 0xf5, 0x79, 0x74],
);

/// `NV_ENC_CAPS_PARAM`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct NV_ENC_CAPS_PARAM {
    pub(crate) version: u32,
    pub(crate) capsToQuery: NV_ENC_CAPS,
    pub(crate) reserved: [u32; 62],
}

/// `NV_ENC_CREATE_INPUT_BUFFER`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct NV_ENC_CREATE_INPUT_BUFFER {
    pub(crate) version: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) memoryHeap: NvEncEnum,
    pub(crate) bufferFmt: NV_ENC_BUFFER_FORMAT,
    pub(crate) reserved: u32,
    pub(crate) inputBuffer: *mut c_void,
    pub(crate) pSysMemBuffer: *mut c_void,
    pub(crate) reserved1: [u32; 58],
    pub(crate) reserved2: [*mut c_void; 63],
}

/// `NV_ENC_CREATE_BITSTREAM_BUFFER`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct NV_ENC_CREATE_BITSTREAM_BUFFER {
    pub(crate) version: u32,
    pub(crate) size: u32,
    pub(crate) memoryHeap: NvEncEnum,
    pub(crate) reserved: u32,
    pub(crate) bitstreamBuffer: *mut c_void,
    pub(crate) bitstreamBufferPtr: *mut c_void,
    pub(crate) reserved1: [u32; 58],
    pub(crate) reserved2: [*mut c_void; 64],
}

/// `NV_ENC_QP`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct NV_ENC_QP {
    pub(crate) qpInterP: u32,
    pub(crate) qpInterB: u32,
    pub(crate) qpIntra: u32,
}

/// `NV_ENC_RC_PARAMS`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct NV_ENC_RC_PARAMS {
    pub(crate) version: u32,
    pub(crate) rateControlMode: NV_ENC_PARAMS_RC_MODE,
    pub(crate) constQP: NV_ENC_QP,
    pub(crate) averageBitRate: u32,
    pub(crate) maxBitRate: u32,
    pub(crate) vbvBufferSize: u32,
    pub(crate) vbvInitialDelay: u32,
    /// `enableMinQP` to `reservedBitFields`: [`bits::rc`]. Name is ours.
    pub(crate) flags: u32,
    pub(crate) minQP: NV_ENC_QP,
    pub(crate) maxQP: NV_ENC_QP,
    pub(crate) initialRCQP: NV_ENC_QP,
    pub(crate) temporallayerIdxMask: u32,
    pub(crate) temporalLayerQP: [u8; 8],
    pub(crate) targetQuality: u8,
    pub(crate) targetQualityLSB: u8,
    pub(crate) lookaheadDepth: u16,
    pub(crate) lowDelayKeyFrameScale: u8,
    pub(crate) yDcQPIndexOffset: i8,
    pub(crate) uDcQPIndexOffset: i8,
    pub(crate) vDcQPIndexOffset: i8,
    pub(crate) qpMapMode: NvEncEnum,
    pub(crate) multiPass: NvEncEnum,
    pub(crate) alphaLayerBitrateRatio: u32,
    pub(crate) cbQPIndexOffset: i8,
    pub(crate) crQPIndexOffset: i8,
    pub(crate) reserved2: u16,
    pub(crate) lookaheadLevel: NvEncEnum,
    pub(crate) viewBitrateRatios: [u8; 7],
    pub(crate) reserved3: u8,
    pub(crate) reserved1: u32,
}

/// `NV_ENC_CONFIG_H264_VUI_PARAMETERS`, which HEVC shares.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct NV_ENC_CONFIG_H264_VUI_PARAMETERS {
    pub(crate) overscanInfoPresentFlag: u32,
    pub(crate) overscanInfo: u32,
    pub(crate) videoSignalTypePresentFlag: u32,
    pub(crate) videoFormat: NvEncEnum,
    pub(crate) videoFullRangeFlag: u32,
    pub(crate) colourDescriptionPresentFlag: u32,
    pub(crate) colourPrimaries: NvEncEnum,
    pub(crate) transferCharacteristics: NvEncEnum,
    pub(crate) colourMatrix: NvEncEnum,
    pub(crate) chromaSampleLocationFlag: u32,
    pub(crate) chromaSampleLocationTop: u32,
    pub(crate) chromaSampleLocationBot: u32,
    pub(crate) bitstreamRestrictionFlag: u32,
    pub(crate) timingInfoPresentFlag: u32,
    pub(crate) numUnitInTicks: u32,
    pub(crate) timeScale: u32,
    pub(crate) reserved: [u32; 12],
}

/// `NV_ENC_VUI_VIDEO_FORMAT_UNSPECIFIED`.
pub(crate) const NV_ENC_VUI_VIDEO_FORMAT_UNSPECIFIED: NvEncEnum = 5;

/// `NV_ENC_CONFIG_H264`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct NV_ENC_CONFIG_H264 {
    /// `enableTemporalSVC` to `reservedBitFields`: [`bits::h264`]. Name is
    /// ours.
    pub(crate) flags: u32,
    pub(crate) level: u32,
    pub(crate) idrPeriod: u32,
    pub(crate) separateColourPlaneFlag: u32,
    pub(crate) disableDeblockingFilterIDC: u32,
    pub(crate) numTemporalLayers: u32,
    pub(crate) spsId: u32,
    pub(crate) ppsId: u32,
    pub(crate) adaptiveTransformMode: NvEncEnum,
    pub(crate) fmoMode: NvEncEnum,
    pub(crate) bdirectMode: NvEncEnum,
    pub(crate) entropyCodingMode: NvEncEnum,
    pub(crate) stereoMode: NvEncEnum,
    pub(crate) intraRefreshPeriod: u32,
    pub(crate) intraRefreshCnt: u32,
    pub(crate) maxNumRefFrames: u32,
    pub(crate) sliceMode: u32,
    pub(crate) sliceModeData: u32,
    pub(crate) h264VUIParameters: NV_ENC_CONFIG_H264_VUI_PARAMETERS,
    pub(crate) ltrNumFrames: u32,
    pub(crate) ltrTrustMode: u32,
    pub(crate) chromaFormatIDC: u32,
    pub(crate) maxTemporalLayers: u32,
    pub(crate) useBFramesAsRef: NvEncEnum,
    pub(crate) numRefL0: NvEncEnum,
    pub(crate) numRefL1: NvEncEnum,
    pub(crate) outputBitDepth: NvEncEnum,
    pub(crate) inputBitDepth: NvEncEnum,
    pub(crate) tfLevel: NvEncEnum,
    pub(crate) reserved1: [u32; 264],
    pub(crate) reserved2: [*mut c_void; 64],
}

/// `NV_ENC_CONFIG_HEVC`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct NV_ENC_CONFIG_HEVC {
    pub(crate) level: u32,
    pub(crate) tier: u32,
    pub(crate) minCUSize: NvEncEnum,
    pub(crate) maxCUSize: NvEncEnum,
    /// `useConstrainedIntraPred` to `reserved`: [`bits::hevc`]. Name is ours.
    pub(crate) flags: u32,
    pub(crate) idrPeriod: u32,
    pub(crate) intraRefreshPeriod: u32,
    pub(crate) intraRefreshCnt: u32,
    pub(crate) maxNumRefFramesInDPB: u32,
    pub(crate) ltrNumFrames: u32,
    pub(crate) vpsId: u32,
    pub(crate) spsId: u32,
    pub(crate) ppsId: u32,
    pub(crate) sliceMode: u32,
    pub(crate) sliceModeData: u32,
    pub(crate) maxTemporalLayersMinus1: u32,
    pub(crate) hevcVUIParameters: NV_ENC_CONFIG_H264_VUI_PARAMETERS,
    pub(crate) ltrTrustMode: u32,
    pub(crate) useBFramesAsRef: NvEncEnum,
    pub(crate) numRefL0: NvEncEnum,
    pub(crate) numRefL1: NvEncEnum,
    pub(crate) tfLevel: NvEncEnum,
    pub(crate) disableDeblockingFilterIDC: u32,
    pub(crate) outputBitDepth: NvEncEnum,
    pub(crate) inputBitDepth: NvEncEnum,
    pub(crate) numTemporalLayers: u32,
    pub(crate) numViews: u32,
    pub(crate) reserved1: [u32; 208],
    pub(crate) reserved2: [*mut c_void; 64],
}

/// `NV_ENC_CODEC_CONFIG`. AV1's and the motion-estimation configurations
/// are the union's other members; `reserved` holds its whole size.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) union NV_ENC_CODEC_CONFIG {
    pub(crate) h264Config: NV_ENC_CONFIG_H264,
    pub(crate) hevcConfig: NV_ENC_CONFIG_HEVC,
    pub(crate) reserved: [u64; 224],
}

/// `NV_ENC_CONFIG`.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct NV_ENC_CONFIG {
    pub(crate) version: u32,
    pub(crate) profileGUID: GUID,
    pub(crate) gopLength: u32,
    pub(crate) frameIntervalP: i32,
    pub(crate) monoChromeEncoding: u32,
    pub(crate) frameFieldMode: NvEncEnum,
    pub(crate) mvPrecision: NvEncEnum,
    pub(crate) rcParams: NV_ENC_RC_PARAMS,
    pub(crate) encodeCodecConfig: NV_ENC_CODEC_CONFIG,
    pub(crate) reserved: [u32; 278],
    pub(crate) reserved2: [*mut c_void; 64],
}

/// `NVENC_EXTERNAL_ME_HINT_COUNTS_PER_BLOCKTYPE`, never set here.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct NVENC_EXTERNAL_ME_HINT_COUNTS_PER_BLOCKTYPE {
    pub(crate) counts: u32,
    pub(crate) reserved1: [u32; 3],
}

/// `NV_ENC_INITIALIZE_PARAMS`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct NV_ENC_INITIALIZE_PARAMS {
    pub(crate) version: u32,
    pub(crate) encodeGUID: GUID,
    pub(crate) presetGUID: GUID,
    pub(crate) encodeWidth: u32,
    pub(crate) encodeHeight: u32,
    pub(crate) darWidth: u32,
    pub(crate) darHeight: u32,
    pub(crate) frameRateNum: u32,
    pub(crate) frameRateDen: u32,
    pub(crate) enableEncodeAsync: u32,
    pub(crate) enablePTD: u32,
    /// `reportSliceOffsets` to `reservedBitFields`: never set here. Name is
    /// ours.
    pub(crate) flags: u32,
    pub(crate) privDataSize: u32,
    pub(crate) reserved: u32,
    pub(crate) privData: *mut c_void,
    pub(crate) encodeConfig: *mut NV_ENC_CONFIG,
    pub(crate) maxEncodeWidth: u32,
    pub(crate) maxEncodeHeight: u32,
    pub(crate) maxMEHintCountsPerBlock: [NVENC_EXTERNAL_ME_HINT_COUNTS_PER_BLOCKTYPE; 2],
    pub(crate) tuningInfo: NV_ENC_TUNING_INFO,
    pub(crate) bufferFormat: NV_ENC_BUFFER_FORMAT,
    pub(crate) numStateBuffers: u32,
    pub(crate) outputStatsLevel: NvEncEnum,
    pub(crate) reserved1: [u32; 284],
    pub(crate) reserved2: [*mut c_void; 64],
}

/// `NV_ENC_RECONFIGURE_PARAMS`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct NV_ENC_RECONFIGURE_PARAMS {
    pub(crate) version: u32,
    pub(crate) reserved: u32,
    pub(crate) reInitEncodeParams: NV_ENC_INITIALIZE_PARAMS,
    /// `resetEncoder : 1`, `forceIDR : 1`, `reserved : 30`: never set
    /// here. Name is ours.
    pub(crate) flags: u32,
    pub(crate) reserved2: u32,
}

/// `NV_ENC_PRESET_CONFIG`.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct NV_ENC_PRESET_CONFIG {
    pub(crate) version: u32,
    pub(crate) reserved: u32,
    pub(crate) presetCfg: NV_ENC_CONFIG,
    pub(crate) reserved1: [u32; 256],
    pub(crate) reserved2: [*mut c_void; 64],
}

/// `NV_ENC_CODEC_PIC_PARAMS`: the codecs' per-picture parameters, which a
/// session with picture type decision leaves zero.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct NV_ENC_CODEC_PIC_PARAMS {
    pub(crate) reserved: [u64; 193],
}

/// `NV_ENC_PIC_PARAMS`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct NV_ENC_PIC_PARAMS {
    pub(crate) version: u32,
    pub(crate) inputWidth: u32,
    pub(crate) inputHeight: u32,
    pub(crate) inputPitch: u32,
    pub(crate) encodePicFlags: u32,
    pub(crate) frameIdx: u32,
    pub(crate) inputTimeStamp: u64,
    pub(crate) inputDuration: u64,
    pub(crate) inputBuffer: *mut c_void,
    pub(crate) outputBitstream: *mut c_void,
    pub(crate) completionEvent: *mut c_void,
    pub(crate) bufferFmt: NV_ENC_BUFFER_FORMAT,
    pub(crate) pictureStruct: NV_ENC_PIC_STRUCT,
    pub(crate) pictureType: NV_ENC_PIC_TYPE,
    pub(crate) codecPicParams: NV_ENC_CODEC_PIC_PARAMS,
    pub(crate) meHintCountsPerBlock: [NVENC_EXTERNAL_ME_HINT_COUNTS_PER_BLOCKTYPE; 2],
    pub(crate) meExternalHints: *mut c_void,
    pub(crate) reserved2: [u32; 7],
    pub(crate) reserved5: [*mut c_void; 2],
    pub(crate) qpDeltaMap: *mut i8,
    pub(crate) qpDeltaMapSize: u32,
    pub(crate) reservedBitFields: u32,
    pub(crate) meHintRefPicDist: [u16; 2],
    pub(crate) diffPicNumHint: i32,
    pub(crate) alphaBuffer: *mut c_void,
    pub(crate) meExternalSbHints: *mut c_void,
    pub(crate) meSbHintsCount: u32,
    pub(crate) stateBufferIdx: u32,
    pub(crate) outputReconBuffer: *mut c_void,
    pub(crate) reserved3: [u32; 284],
    pub(crate) reserved6: [*mut c_void; 57],
}

/// `NV_ENC_LOCK_BITSTREAM`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct NV_ENC_LOCK_BITSTREAM {
    pub(crate) version: u32,
    /// `doNotWait`, `ltrFrame`, `getRCStats`, `reservedBitFields`. Name is
    /// ours.
    pub(crate) flags: u32,
    pub(crate) outputBitstream: *mut c_void,
    pub(crate) sliceOffsets: *mut u32,
    pub(crate) frameIdx: u32,
    pub(crate) hwEncodeStatus: u32,
    pub(crate) numSlices: u32,
    pub(crate) bitstreamSizeInBytes: u32,
    pub(crate) outputTimeStamp: u64,
    pub(crate) outputDuration: u64,
    pub(crate) bitstreamBufferPtr: *mut c_void,
    pub(crate) pictureType: NV_ENC_PIC_TYPE,
    pub(crate) pictureStruct: NV_ENC_PIC_STRUCT,
    pub(crate) frameAvgQP: u32,
    pub(crate) frameSatd: u32,
    pub(crate) ltrFrameIdx: u32,
    pub(crate) ltrFrameBitmap: u32,
    pub(crate) temporalId: u32,
    pub(crate) intraMBCount: u32,
    pub(crate) interMBCount: u32,
    pub(crate) averageMVX: i32,
    pub(crate) averageMVY: i32,
    pub(crate) alphaLayerSizeInBytes: u32,
    pub(crate) outputStatsPtrSize: u32,
    pub(crate) reserved: u32,
    pub(crate) outputStatsPtr: *mut c_void,
    pub(crate) frameIdxDisplay: u32,
    pub(crate) reserved1: [u32; 219],
    pub(crate) reserved2: [*mut c_void; 63],
    pub(crate) reservedInternal: [u32; 8],
}

/// `NV_ENC_LOCK_INPUT_BUFFER`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct NV_ENC_LOCK_INPUT_BUFFER {
    pub(crate) version: u32,
    /// `doNotWait : 1`, `reservedBitFields : 31`. Name is ours.
    pub(crate) flags: u32,
    pub(crate) inputBuffer: *mut c_void,
    pub(crate) bufferDataPtr: *mut c_void,
    pub(crate) pitch: u32,
    pub(crate) reserved1: [u32; 251],
    pub(crate) reserved2: [*mut c_void; 64],
}

/// `NV_ENC_SEQUENCE_PARAM_PAYLOAD`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct NV_ENC_SEQUENCE_PARAM_PAYLOAD {
    pub(crate) version: u32,
    pub(crate) inBufferSize: u32,
    pub(crate) spsId: u32,
    pub(crate) ppsId: u32,
    pub(crate) spsppsBuffer: *mut c_void,
    pub(crate) outSPSPPSPayloadSize: *mut u32,
    pub(crate) reserved: [u32; 250],
    pub(crate) reserved2: [*mut c_void; 64],
}

/// `NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS {
    pub(crate) version: u32,
    pub(crate) deviceType: NvEncEnum,
    pub(crate) device: *mut c_void,
    pub(crate) reserved: *mut c_void,
    pub(crate) apiVersion: u32,
    pub(crate) reserved1: [u32; 253],
    pub(crate) reserved2: [*mut c_void; 64],
}

/// The positions of the C bit-fields this crate sets, in the `u32` words
/// that hold them: GCC's x86-64 layout, from the least significant bit up
/// (checked by `crate::nvidia::layout`'s test).
pub(crate) mod bits {
    pub(crate) use crate::nvidia::nvdec::ffi::bits::Bits;

    const fn b(shift: u32, width: u32) -> Bits {
        Bits { shift, width }
    }

    /// `NV_ENC_RC_PARAMS`'s flags.
    pub(crate) mod rc {
        use super::{Bits, b};
        pub(crate) const enableLookahead: Bits = b(5, 1);
    }

    /// `NV_ENC_CONFIG_H264`'s flags.
    pub(crate) mod h264 {
        use super::{Bits, b};
        pub(crate) const repeatSPSPPS: Bits = b(12, 1);
    }

    /// `NV_ENC_CONFIG_HEVC`'s flags.
    pub(crate) mod hevc {
        use super::{Bits, b};
        pub(crate) const repeatSPSPPS: Bits = b(7, 1);
        pub(crate) const chromaFormatIDC: Bits = b(9, 2);
    }

    /// `NV_ENC_LOCK_BITSTREAM`'s flags: `doNotWait`, which only the probe
    /// of the lock sets (`nvenc::locks`).
    #[cfg(test)]
    pub(crate) mod lock {
        use super::{Bits, b};
        pub(crate) const doNotWait: Bits = b(0, 1);
    }
}

/// Declares `NV_ENCODE_API_FUNCTION_LIST`: its version and reserved words,
/// then each entry point in the header's order, then its reserved tail.
/// Entries are `Option`, null where the driver has none. `reserved1`, a
/// `void*` between two entry points, is declared as one so that the order
/// holds; it is never called.
macro_rules! function_list {
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident {
            $( $field:ident: fn($($arg:ty),* $(,)?) -> $ret:ty; )*
        }
        reserved2: $count:literal
    ) => {
        $(#[$meta])*
        #[repr(C)]
        #[derive(Clone, Copy)]
        $vis struct $name {
            pub(crate) version: u32,
            pub(crate) reserved: u32,
            $( pub(crate) $field: Option<unsafe extern "C" fn($($arg),*) -> $ret>, )*
            pub(crate) reserved2: [*mut c_void; $count],
        }

        #[cfg(test)]
        impl $name {
            /// Each entry's C signature as declared here, for the check
            /// against the header.
            pub(crate) const SIGNATURES: &'static [crate::nvidia::dl::Signature] = &[
                $(
                    crate::nvidia::dl::Signature {
                        symbol: stringify!($field),
                        result: stringify!($ret),
                        arguments: &[$(stringify!($arg)),*],
                    },
                )*
            ];
        }
    };
}

/// The opaque encoder handle every entry point takes. Name is ours: the
/// header's `void* encoder`.
pub(crate) type Encoder = *mut c_void;

function_list! {
    /// `NV_ENCODE_API_FUNCTION_LIST`, which `NvEncodeAPICreateInstance`
    /// fills.
    pub(crate) struct NV_ENCODE_API_FUNCTION_LIST {
        nvEncOpenEncodeSession: fn(*mut c_void, u32, *mut Encoder) -> NVENCSTATUS;
        nvEncGetEncodeGUIDCount: fn(Encoder, *mut u32) -> NVENCSTATUS;
        nvEncGetEncodeProfileGUIDCount: fn(Encoder, GUID, *mut u32) -> NVENCSTATUS;
        nvEncGetEncodeProfileGUIDs: fn(Encoder, GUID, *mut GUID, u32, *mut u32) -> NVENCSTATUS;
        nvEncGetEncodeGUIDs: fn(Encoder, *mut GUID, u32, *mut u32) -> NVENCSTATUS;
        nvEncGetInputFormatCount: fn(Encoder, GUID, *mut u32) -> NVENCSTATUS;
        nvEncGetInputFormats: fn(Encoder, GUID, *mut NV_ENC_BUFFER_FORMAT, u32, *mut u32)
            -> NVENCSTATUS;
        nvEncGetEncodeCaps: fn(Encoder, GUID, *mut NV_ENC_CAPS_PARAM, *mut c_int) -> NVENCSTATUS;
        nvEncGetEncodePresetCount: fn(Encoder, GUID, *mut u32) -> NVENCSTATUS;
        nvEncGetEncodePresetGUIDs: fn(Encoder, GUID, *mut GUID, u32, *mut u32) -> NVENCSTATUS;
        nvEncGetEncodePresetConfig: fn(Encoder, GUID, GUID, *mut NV_ENC_PRESET_CONFIG)
            -> NVENCSTATUS;
        nvEncInitializeEncoder: fn(Encoder, *mut NV_ENC_INITIALIZE_PARAMS) -> NVENCSTATUS;
        nvEncCreateInputBuffer: fn(Encoder, *mut NV_ENC_CREATE_INPUT_BUFFER) -> NVENCSTATUS;
        nvEncDestroyInputBuffer: fn(Encoder, *mut c_void) -> NVENCSTATUS;
        nvEncCreateBitstreamBuffer: fn(Encoder, *mut NV_ENC_CREATE_BITSTREAM_BUFFER)
            -> NVENCSTATUS;
        nvEncDestroyBitstreamBuffer: fn(Encoder, *mut c_void) -> NVENCSTATUS;
        nvEncEncodePicture: fn(Encoder, *mut NV_ENC_PIC_PARAMS) -> NVENCSTATUS;
        nvEncLockBitstream: fn(Encoder, *mut NV_ENC_LOCK_BITSTREAM) -> NVENCSTATUS;
        nvEncUnlockBitstream: fn(Encoder, *mut c_void) -> NVENCSTATUS;
        nvEncLockInputBuffer: fn(Encoder, *mut NV_ENC_LOCK_INPUT_BUFFER) -> NVENCSTATUS;
        nvEncUnlockInputBuffer: fn(Encoder, *mut c_void) -> NVENCSTATUS;
        nvEncGetEncodeStats: fn(Encoder, *mut c_void) -> NVENCSTATUS;
        nvEncGetSequenceParams: fn(Encoder, *mut NV_ENC_SEQUENCE_PARAM_PAYLOAD) -> NVENCSTATUS;
        nvEncRegisterAsyncEvent: fn(Encoder, *mut c_void) -> NVENCSTATUS;
        nvEncUnregisterAsyncEvent: fn(Encoder, *mut c_void) -> NVENCSTATUS;
        nvEncMapInputResource: fn(Encoder, *mut c_void) -> NVENCSTATUS;
        nvEncUnmapInputResource: fn(Encoder, *mut c_void) -> NVENCSTATUS;
        nvEncDestroyEncoder: fn(Encoder) -> NVENCSTATUS;
        nvEncInvalidateRefFrames: fn(Encoder, u64) -> NVENCSTATUS;
        nvEncOpenEncodeSessionEx: fn(*mut NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS, *mut Encoder)
            -> NVENCSTATUS;
        nvEncRegisterResource: fn(Encoder, *mut c_void) -> NVENCSTATUS;
        nvEncUnregisterResource: fn(Encoder, *mut c_void) -> NVENCSTATUS;
        nvEncReconfigureEncoder: fn(Encoder, *mut NV_ENC_RECONFIGURE_PARAMS) -> NVENCSTATUS;
        reserved1: fn() -> NVENCSTATUS;
        nvEncCreateMVBuffer: fn(Encoder, *mut c_void) -> NVENCSTATUS;
        nvEncDestroyMVBuffer: fn(Encoder, *mut c_void) -> NVENCSTATUS;
        nvEncRunMotionEstimationOnly: fn(Encoder, *mut c_void) -> NVENCSTATUS;
        nvEncGetLastErrorString: fn(Encoder) -> *const c_char;
        nvEncSetIOCudaStreams: fn(Encoder, *mut c_void, *mut c_void) -> NVENCSTATUS;
        nvEncGetEncodePresetConfigEx: fn(
            Encoder,
            GUID,
            GUID,
            NV_ENC_TUNING_INFO,
            *mut NV_ENC_PRESET_CONFIG,
        ) -> NVENCSTATUS;
        nvEncGetSequenceParamEx: fn(
            Encoder,
            *mut NV_ENC_INITIALIZE_PARAMS,
            *mut NV_ENC_SEQUENCE_PARAM_PAYLOAD,
        ) -> NVENCSTATUS;
        nvEncRestoreEncoderState: fn(Encoder, *mut c_void) -> NVENCSTATUS;
        nvEncLookaheadPicture: fn(Encoder, *mut c_void) -> NVENCSTATUS;
    }
    reserved2: 275
}

crate::nvidia::dl::functions! {
    /// libnvidia-encode's two exported entry points.
    pub(crate) struct NvencEntryPoints {
        NvEncodeAPICreateInstance: fn(*mut NV_ENCODE_API_FUNCTION_LIST) -> NVENCSTATUS
            = c"NvEncodeAPICreateInstance";
        NvEncodeAPIGetMaxSupportedVersion: fn(*mut u32) -> NVENCSTATUS
            = c"NvEncodeAPIGetMaxSupportedVersion";
    }
}

// SAFETY: each is repr(C) and holds only integers, raw pointers, `Option`s
// of function pointers, and arrays, structs and unions of those
// (`Zeroed`'s contract).
unsafe impl Zeroed for NV_ENC_CAPS_PARAM {}
// SAFETY: as above.
unsafe impl Zeroed for NV_ENC_CREATE_INPUT_BUFFER {}
// SAFETY: as above.
unsafe impl Zeroed for NV_ENC_CREATE_BITSTREAM_BUFFER {}
// SAFETY: as above.
unsafe impl Zeroed for NV_ENC_CONFIG {}
// SAFETY: as above.
unsafe impl Zeroed for NV_ENC_INITIALIZE_PARAMS {}
// SAFETY: as above.
unsafe impl Zeroed for NV_ENC_RECONFIGURE_PARAMS {}
// SAFETY: as above.
unsafe impl Zeroed for NV_ENC_PRESET_CONFIG {}
// SAFETY: as above.
unsafe impl Zeroed for NV_ENC_PIC_PARAMS {}
// SAFETY: as above.
unsafe impl Zeroed for NV_ENC_LOCK_BITSTREAM {}
// SAFETY: as above.
unsafe impl Zeroed for NV_ENC_LOCK_INPUT_BUFFER {}
// SAFETY: as above.
unsafe impl Zeroed for NV_ENC_SEQUENCE_PARAM_PAYLOAD {}
// SAFETY: as above.
unsafe impl Zeroed for NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS {}
// SAFETY: as above.
unsafe impl Zeroed for NV_ENCODE_API_FUNCTION_LIST {}

// The sizes ffnvcodec 13.1.15.0's header gives on x86-64 Linux
// (2026-10-24); `crate::nvidia::layout`'s test checks every offset and bit-field
// too.
#[cfg(target_pointer_width = "64")]
const _: () = {
    assert!(size_of::<NV_ENC_CAPS_PARAM>() == 256);
    assert!(size_of::<NV_ENC_CREATE_INPUT_BUFFER>() == 776);
    assert!(size_of::<NV_ENC_CREATE_BITSTREAM_BUFFER>() == 776);
    assert!(size_of::<NV_ENC_RC_PARAMS>() == 128);
    assert!(size_of::<NV_ENC_CONFIG_H264>() == 1792);
    assert!(size_of::<NV_ENC_CONFIG_HEVC>() == 1560);
    assert!(size_of::<NV_ENC_CODEC_CONFIG>() == 1792);
    assert!(size_of::<NV_ENC_CONFIG>() == 3584);
    assert!(size_of::<NV_ENC_INITIALIZE_PARAMS>() == 1800);
    assert!(size_of::<NV_ENC_RECONFIGURE_PARAMS>() == 1816);
    assert!(size_of::<NV_ENC_PRESET_CONFIG>() == 5128);
    assert!(size_of::<NV_ENC_CODEC_PIC_PARAMS>() == 1544);
    assert!(size_of::<NV_ENC_PIC_PARAMS>() == 3360);
    assert!(size_of::<NV_ENC_LOCK_BITSTREAM>() == 1544);
    assert!(size_of::<NV_ENC_LOCK_INPUT_BUFFER>() == 1544);
    assert!(size_of::<NV_ENC_SEQUENCE_PARAM_PAYLOAD>() == 1544);
    assert!(size_of::<NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS>() == 1552);
    assert!(size_of::<NV_ENCODE_API_FUNCTION_LIST>() == 2552);
};
