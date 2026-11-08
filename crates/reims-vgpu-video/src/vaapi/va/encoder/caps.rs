//! What a driver says one encoder can do: the attributes of its profile and
//! encode entrypoint that decide what the encoder may write (va.h,
//! `VAConfigAttribType`; va_enc_hevc.h for HEVC's features and block sizes).

use super::super::caps::{Entrypoint, Profile};
use super::super::display::{Display, Error};
use super::super::ffi::{self, encode as enc};

/// A `VA_FEATURE_*` answer for one HEVC tool (va.h).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Feature {
    /// `VA_FEATURE_NOT_SUPPORTED`: the flag must be 0.
    Unsupported,
    /// `VA_FEATURE_SUPPORTED`: the encoder's choice.
    Supported,
    /// `VA_FEATURE_REQUIRED`: the flag must be 1.
    Required,
}

impl Feature {
    fn from_bits(value: u32, shift: u32) -> Self {
        match (value >> shift) & 3 {
            0 => Self::Unsupported,
            2 => Self::Required,
            _ => Self::Supported,
        }
    }

    /// The flag to write when the encoder would like `wanted`: what the
    /// driver requires, else `wanted` where the driver allows it.
    pub fn resolve(self, wanted: bool) -> bool {
        match self {
            Self::Unsupported => false,
            Self::Supported => wanted,
            Self::Required => true,
        }
    }
}

/// `VAConfigAttribValEncHEVCFeatures`, field by field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[expect(missing_docs, reason = "each field is va_enc_hevc.h's, by its name")]
pub struct HevcFeatures {
    pub separate_colour_planes: Feature,
    pub scaling_lists: Feature,
    pub amp: Feature,
    pub sao: Feature,
    pub pcm: Feature,
    pub temporal_mvp: Feature,
    pub strong_intra_smoothing: Feature,
    pub dependent_slices: Feature,
    pub sign_data_hiding: Feature,
    pub constrained_intra_pred: Feature,
    pub transform_skip: Feature,
    pub cu_qp_delta: Feature,
    pub weighted_prediction: Feature,
    pub transquant_bypass: Feature,
    pub deblocking_filter_disable: Feature,
}

impl HevcFeatures {
    /// Reads the attribute's value.
    pub fn from_value(value: u32) -> Self {
        let f = |index: u32| Feature::from_bits(value, index * 2);
        Self {
            separate_colour_planes: f(0),
            scaling_lists: f(1),
            amp: f(2),
            sao: f(3),
            pcm: f(4),
            temporal_mvp: f(5),
            strong_intra_smoothing: f(6),
            dependent_slices: f(7),
            sign_data_hiding: f(8),
            constrained_intra_pred: f(9),
            transform_skip: f(10),
            cu_qp_delta: f(11),
            weighted_prediction: f(12),
            transquant_bypass: f(13),
            deblocking_filter_disable: f(14),
        }
    }
}

/// `VAConfigAttribValEncHEVCBlockSizes`, field by field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[expect(missing_docs, reason = "each field is va_enc_hevc.h's, by its name")]
pub struct HevcBlockSizes {
    pub log2_max_coding_tree_block_size_minus3: u8,
    pub log2_min_coding_tree_block_size_minus3: u8,
    pub log2_min_luma_coding_block_size_minus3: u8,
    pub log2_max_luma_transform_block_size_minus2: u8,
    pub log2_min_luma_transform_block_size_minus2: u8,
    pub max_max_transform_hierarchy_depth_inter: u8,
    pub min_max_transform_hierarchy_depth_inter: u8,
    pub max_max_transform_hierarchy_depth_intra: u8,
    pub min_max_transform_hierarchy_depth_intra: u8,
    pub log2_max_pcm_coding_block_size_minus3: u8,
    pub log2_min_pcm_coding_block_size_minus3: u8,
}

impl HevcBlockSizes {
    /// Reads the attribute's value.
    pub fn from_value(value: u32) -> Self {
        let f = |index: u32| u8::try_from((value >> (index * 2)) & 3).unwrap_or(0);
        Self {
            log2_max_coding_tree_block_size_minus3: f(0),
            log2_min_coding_tree_block_size_minus3: f(1),
            log2_min_luma_coding_block_size_minus3: f(2),
            log2_max_luma_transform_block_size_minus2: f(3),
            log2_min_luma_transform_block_size_minus2: f(4),
            max_max_transform_hierarchy_depth_inter: f(5),
            min_max_transform_hierarchy_depth_inter: f(6),
            max_max_transform_hierarchy_depth_intra: f(7),
            min_max_transform_hierarchy_depth_intra: f(8),
            log2_max_pcm_coding_block_size_minus3: f(9),
            log2_min_pcm_coding_block_size_minus3: f(10),
        }
    }
}

/// What the driver reports for one encode profile and entrypoint. Each
/// attribute is `None` where the driver marks it unsupported.
#[derive(Clone, Debug)]
pub struct EncoderCapabilities {
    /// The VA profile.
    pub profile: Profile,
    /// The encode entrypoint (`EncSlice` or `EncSliceLP`).
    pub entrypoint: Entrypoint,
    /// `VAConfigAttribRTFormat`: `VA_RT_FORMAT_*` bits.
    pub rt_formats: Option<u32>,
    /// `VAConfigAttribRateControl`: `VA_RC_*` bits.
    pub rate_control: Option<u32>,
    /// `VAConfigAttribEncPackedHeaders`: `VA_ENC_PACKED_HEADER_*` bits.
    pub packed_headers: Option<u32>,
    /// `VAConfigAttribEncMaxRefFrames`: list 0 in the low 16 bits, list 1
    /// in the high.
    pub max_ref_frames: Option<u32>,
    /// `VAConfigAttribEncMaxSlices`.
    pub max_slices: Option<u32>,
    /// `VAConfigAttribEncSliceStructure`: `VA_ENC_SLICE_STRUCTURE_*` bits.
    pub slice_structure: Option<u32>,
    /// `VAConfigAttribEncQualityRange`: the lowest quality level (1 is the
    /// best).
    pub quality_range: Option<u32>,
    /// `VAConfigAttribEncIntraRefresh`.
    pub intra_refresh: Option<u32>,
    /// `VAConfigAttribEncSkipFrame`.
    pub skip_frame: Option<u32>,
    /// `VAConfigAttribEncROI`.
    pub roi: Option<u32>,
    /// `VAConfigAttribEncRateControlExt`.
    pub rate_control_ext: Option<u32>,
    /// `VAConfigAttribMaxFrameSize`.
    pub max_frame_size: Option<u32>,
    /// `VAConfigAttribPredictionDirection`: `VA_PREDICTION_DIRECTION_*`.
    pub prediction_direction: Option<u32>,
    /// `VAConfigAttribMaxPictureWidth`.
    pub max_width: Option<u32>,
    /// `VAConfigAttribMaxPictureHeight`.
    pub max_height: Option<u32>,
    /// `VAConfigAttribEncHEVCFeatures`, raw.
    pub hevc_features: Option<u32>,
    /// `VAConfigAttribEncHEVCBlockSizes`, raw.
    pub hevc_block_sizes: Option<u32>,
}

impl EncoderCapabilities {
    /// The pair's attributes, or `None` when the driver lacks the pair.
    pub fn query(
        display: &Display,
        profile: Profile,
        entrypoint: Entrypoint,
    ) -> Result<Option<Self>, Error> {
        if !display.supports(profile, entrypoint) {
            return Ok(None);
        }
        let types = [
            ffi::VAConfigAttribRTFormat,
            ffi::VAConfigAttribRateControl,
            ffi::VAConfigAttribEncPackedHeaders,
            ffi::VAConfigAttribEncMaxRefFrames,
            ffi::VAConfigAttribEncMaxSlices,
            enc::VAConfigAttribEncSliceStructure,
            enc::VAConfigAttribEncQualityRange,
            enc::VAConfigAttribEncIntraRefresh,
            enc::VAConfigAttribEncSkipFrame,
            enc::VAConfigAttribEncROI,
            enc::VAConfigAttribEncRateControlExt,
            enc::VAConfigAttribMaxFrameSize,
            enc::VAConfigAttribPredictionDirection,
            ffi::VAConfigAttribMaxPictureWidth,
            ffi::VAConfigAttribMaxPictureHeight,
            enc::VAConfigAttribEncHEVCFeatures,
            enc::VAConfigAttribEncHEVCBlockSizes,
        ];
        let v = display.config_attributes(profile, entrypoint, &types)?;
        Ok(Some(Self {
            profile,
            entrypoint,
            rt_formats: v[0],
            rate_control: v[1],
            packed_headers: v[2],
            max_ref_frames: v[3],
            max_slices: v[4],
            slice_structure: v[5],
            quality_range: v[6],
            intra_refresh: v[7],
            skip_frame: v[8],
            roi: v[9],
            rate_control_ext: v[10],
            max_frame_size: v[11],
            prediction_direction: v[12],
            max_width: v[13],
            max_height: v[14],
            hevc_features: v[15],
            hevc_block_sizes: v[16],
        }))
    }

    /// References the driver allows in list 0 and in list 1.
    pub fn max_references(&self) -> (u32, u32) {
        let value = self.max_ref_frames.unwrap_or(1);
        (value & 0xffff, value >> 16)
    }

    /// HEVC's tools, decoded.
    pub fn hevc_feature_set(&self) -> Option<HevcFeatures> {
        self.hevc_features.map(HevcFeatures::from_value)
    }

    /// HEVC's block sizes, decoded.
    pub fn hevc_block_size_set(&self) -> Option<HevcBlockSizes> {
        self.hevc_block_sizes.map(HevcBlockSizes::from_value)
    }
}
