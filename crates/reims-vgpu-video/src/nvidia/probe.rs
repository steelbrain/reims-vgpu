//! What the GPU's driver answers, in words: NVENC's codecs, profiles,
//! presets, input formats and every capability. `nvidia-probe` prints it.

use std::sync::Arc;

use crate::nvidia::cuda::Gpu;
use crate::nvidia::error::Error;
use crate::nvidia::nvenc::ffi::{self, GUID};
use crate::nvidia::nvenc::{Session, max_api_version};

/// The GUIDs this binding knows, by name.
const GUID_NAMES: [(GUID, &str); 21] = [
    (ffi::NV_ENC_CODEC_H264_GUID, "H264"),
    (ffi::NV_ENC_CODEC_HEVC_GUID, "HEVC"),
    (ffi::NV_ENC_CODEC_AV1_GUID, "AV1"),
    (ffi::NV_ENC_CODEC_PROFILE_AUTOSELECT_GUID, "AUTOSELECT"),
    (ffi::NV_ENC_H264_PROFILE_BASELINE_GUID, "H264_BASELINE"),
    (ffi::NV_ENC_H264_PROFILE_MAIN_GUID, "H264_MAIN"),
    (ffi::NV_ENC_H264_PROFILE_HIGH_GUID, "H264_HIGH"),
    (ffi::NV_ENC_H264_PROFILE_HIGH_10_GUID, "H264_HIGH_10"),
    (ffi::NV_ENC_H264_PROFILE_HIGH_444_GUID, "H264_HIGH_444"),
    (
        ffi::NV_ENC_H264_PROFILE_CONSTRAINED_HIGH_GUID,
        "H264_CONSTRAINED_HIGH",
    ),
    (ffi::NV_ENC_HEVC_PROFILE_MAIN_GUID, "HEVC_MAIN"),
    (ffi::NV_ENC_HEVC_PROFILE_MAIN10_GUID, "HEVC_MAIN10"),
    (ffi::NV_ENC_HEVC_PROFILE_FREXT_GUID, "HEVC_FREXT"),
    (ffi::NV_ENC_AV1_PROFILE_MAIN_GUID, "AV1_MAIN"),
    (ffi::NV_ENC_PRESET_P1_GUID, "P1"),
    (ffi::NV_ENC_PRESET_P2_GUID, "P2"),
    (ffi::NV_ENC_PRESET_P3_GUID, "P3"),
    (ffi::NV_ENC_PRESET_P4_GUID, "P4"),
    (ffi::NV_ENC_PRESET_P5_GUID, "P5"),
    (ffi::NV_ENC_PRESET_P6_GUID, "P6"),
    (ffi::NV_ENC_PRESET_P7_GUID, "P7"),
];

/// A GUID's name where this binding knows it, else its hex.
fn guid_name(guid: GUID) -> String {
    GUID_NAMES
        .iter()
        .find(|(known, _)| *known == guid)
        .map_or_else(
            || {
                let tail: String = guid.Data4.iter().map(|b| format!("{b:02x}")).collect();
                format!(
                    "{:08x}-{:04x}-{:04x}-{tail}",
                    guid.Data1, guid.Data2, guid.Data3
                )
            },
            |(_, name)| (*name).to_owned(),
        )
}

fn buffer_format_name(format: ffi::NV_ENC_BUFFER_FORMAT) -> String {
    ffi::BUFFER_FORMAT_NAMES
        .iter()
        .find(|(value, _)| *value == format)
        .map_or_else(|| format!("{format:#x}"), |(_, name)| (*name).to_owned())
}

/// What NVENC offers for one codec.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodeCodecReport {
    /// The codec.
    pub codec: String,
    /// Its profiles.
    pub profiles: Vec<String>,
    /// Its presets.
    pub presets: Vec<String>,
    /// The input buffer formats it takes.
    pub input_formats: Vec<String>,
    /// Every capability, by name, with the driver's answer.
    pub capabilities: Vec<(&'static str, i32)>,
}

/// What NVENC offers on a GPU.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncoderReport {
    /// `NvEncodeAPIGetMaxSupportedVersion`: major and minor.
    pub max_api_version: (u32, u32),
    /// Each codec the session lists.
    pub codecs: Vec<EncodeCodecReport>,
}

/// Opens an NVENC session on `gpu` and asks it everything.
pub fn encoder_report(gpu: Arc<Gpu>) -> Result<EncoderReport, Error> {
    let session = Session::open(gpu)?;
    let mut codecs = Vec::new();
    for codec in session.codec_guids()? {
        let profiles = session.profile_guids(codec)?;
        let presets = session.preset_guids(codec)?;
        let formats = session.input_formats(codec)?;
        let mut capabilities = Vec::new();
        for (value, name) in ffi::CAPS_NAMES.iter().enumerate() {
            let capability = i32::try_from(value).unwrap_or(0);
            capabilities.push((*name, session.capability(codec, capability)?));
        }
        codecs.push(EncodeCodecReport {
            codec: guid_name(codec),
            profiles: profiles.into_iter().map(guid_name).collect(),
            presets: presets.into_iter().map(guid_name).collect(),
            input_formats: formats.into_iter().map(buffer_format_name).collect(),
            capabilities,
        });
    }
    Ok(EncoderReport {
        max_api_version: max_api_version()?,
        codecs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guids_are_named_or_spelled_out() {
        assert_eq!(guid_name(ffi::NV_ENC_CODEC_HEVC_GUID), "HEVC");
        assert_eq!(
            guid_name(GUID {
                Data1: 1,
                Data2: 2,
                Data3: 3,
                Data4: [4, 5, 6, 7, 8, 9, 10, 11],
            }),
            "00000001-0002-0003-0405060708090a0b"
        );
        assert_eq!(buffer_format_name(0x1_0000), "YUV420_10BIT");
    }
}
